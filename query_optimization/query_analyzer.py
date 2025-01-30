from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sqlparse
import time
import logging
from typing import Dict, List, Tuple

class QueryAnalyzer:
    def __init__(self, connection_string: str):
        """Initialize QueryAnalyzer with database connection string."""
        self.engine = create_engine(connection_string)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('QueryAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def parse_query(self, query: str) -> Dict:
        """Parse SQL query and extract key components."""
        parsed = sqlparse.parse(query)[0]
        
        analysis = {
            'type': str(parsed.get_type()),
            'tables': [],
            'columns': [],
            'where_clauses': [],
            'joins': [],
            'order_by': [],
            'group_by': [],
            'complexity': self._calculate_complexity(parsed)
        }

        for token in parsed.tokens:
            if token.is_group:
                if isinstance(token, sqlparse.sql.Identifier):
                    analysis['tables'].append(str(token))
                elif isinstance(token, sqlparse.sql.Where):
                    analysis['where_clauses'].extend(self._extract_where_conditions(token))

        return analysis

    def _calculate_complexity(self, parsed_query) -> int:
        """Calculate query complexity score."""
        complexity = 0
        for token in parsed_query.tokens:
            if token.is_group:
                complexity += 1
            if str(token).upper() in ['JOIN', 'WHERE', 'GROUP BY', 'ORDER BY']:
                complexity += 2
        return complexity

    def _extract_where_conditions(self, where_clause) -> List[str]:
        """Extract conditions from WHERE clause."""
        conditions = []
        for token in where_clause.tokens:
            if isinstance(token, sqlparse.sql.Comparison):
                conditions.append(str(token))
        return conditions

    def analyze_execution_plan(self, query: str) -> Dict:
        """Generate and analyze query execution plan."""
        try:
            with self.engine.connect() as connection:
                # Different databases have different EXPLAIN syntax
                explain_query = f"EXPLAIN ANALYZE {query}"
                result = connection.execute(text(explain_query))
                plan = result.fetchall()
                
                return self._parse_execution_plan(plan)
        except SQLAlchemyError as e:
            self.logger.error(f"Error analyzing execution plan: {str(e)}")
            return {"error": str(e)}

    def _parse_execution_plan(self, plan_rows: List[Tuple]) -> Dict:
        """Parse and structure the execution plan output."""
        return {
            'steps': [row[0] for row in plan_rows],
            'estimated_cost': self._extract_cost(plan_rows),
            'optimization_suggestions': self._generate_suggestions(plan_rows)
        }

    def _extract_cost(self, plan_rows: List[Tuple]) -> float:
        """Extract cost estimates from execution plan."""
        total_cost = 0
        for row in plan_rows:
            # Extract numerical cost value from plan output
            # This is a simplified version - actual implementation would depend on
            # specific database output format
            cost_str = str(row[0])
            if 'cost=' in cost_str.lower():
                try:
                    cost_part = cost_str.split('cost=')[1].split('..')[1].split()[0]
                    total_cost += float(cost_part)
                except (IndexError, ValueError):
                    continue
        return total_cost

    def _generate_suggestions(self, plan_rows: List[Tuple]) -> List[str]:
        """Generate optimization suggestions based on execution plan."""
        suggestions = []
        plan_text = ' '.join(str(row[0]) for row in plan_rows).lower()

        # Check for common optimization opportunities
        if 'sequential scan' in plan_text:
            suggestions.append("Consider adding an index to avoid sequential scan")
        if 'nested loop' in plan_text:
            suggestions.append("Consider using JOIN hints or rewriting query to avoid nested loops")
        if 'temporary' in plan_text:
            suggestions.append("Query creates temporary tables - consider restructuring")

        return suggestions

    def measure_performance(self, query: str, iterations: int = 3) -> Dict:
        """Measure query performance metrics."""
        timings = []
        
        try:
            with self.engine.connect() as connection:
                for _ in range(iterations):
                    start_time = time.time()
                    connection.execute(text(query))
                    execution_time = time.time() - start_time
                    timings.append(execution_time)

                return {
                    'average_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'iterations': iterations
                }
        except SQLAlchemyError as e:
            self.logger.error(f"Error measuring performance: {str(e)}")
            return {"error": str(e)}

    def optimize_query(self, query: str) -> Dict:
        """Complete query analysis and optimization."""
        results = {
            'original_query': query,
            'parsed_analysis': self.parse_query(query),
            'execution_plan': self.analyze_execution_plan(query),
            'performance_metrics': self.measure_performance(query)
        }

        # Generate optimization recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results

    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []
        
        # Check complexity
        if analysis_results['parsed_analysis']['complexity'] > 5:
            recommendations.append("Query is complex - consider breaking it down into smaller queries")

        # Check performance
        if analysis_results['performance_metrics'].get('average_time', 0) > 1.0:
            recommendations.append("Query execution time is high - consider adding indexes or optimizing JOIN conditions")

        # Add execution plan suggestions
        if 'optimization_suggestions' in analysis_results['execution_plan']:
            recommendations.extend(analysis_results['execution_plan']['optimization_suggestions'])

        return recommendations

def main():
    # Example usage
    connection_string = "postgresql://username:password@localhost:5432/database_name"
    analyzer = QueryAnalyzer(connection_string)
    
    # Example query
    test_query = """
    SELECT users.name, orders.order_date, products.product_name
    FROM users
    JOIN orders ON users.id = orders.user_id
    JOIN products ON orders.product_id = products.id
    WHERE orders.order_date > '2023-01-01'
    GROUP BY users.name, orders.order_date, products.product_name
    ORDER BY orders.order_date DESC;
    """
    
    # Run analysis
    results = analyzer.optimize_query(test_query)
    
    # Print results
    print("\n=== Query Analysis Results ===")
    print("\nParsed Query Analysis:")
    print(f"Query Type: {results['parsed_analysis']['type']}")
    print(f"Complexity Score: {results['parsed_analysis']['complexity']}")
    
    print("\nExecution Plan:")
    for step in results['execution_plan'].get('steps', []):
        print(f"- {step}")
    
    print("\nPerformance Metrics:")
    metrics = results['performance_metrics']
    if 'error' not in metrics:
        print(f"Average Execution Time: {metrics['average_time']:.3f} seconds")
        print(f"Min Execution Time: {metrics['min_time']:.3f} seconds")
        print(f"Max Execution Time: {metrics['max_time']:.3f} seconds")
    
    print("\nOptimization Recommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()