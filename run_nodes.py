import concurrent.futures

def run_all_nodes(nodes, input_data):
    """
    Runs all given perspective nodes on the same input data concurrently.
    Collects and returns the outputs from each perspective.
    """
    results = []
    
    # Use ThreadPoolExecutor to run nodes simultaneously
    # The max_workers will match the number of nodes (e.g., 6)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes) if nodes else 1) as executor:
        # Submit all node processing tasks to the executor
        future_to_node = {executor.submit(node.process, input_data): node for node in nodes}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_node):
            node = future_to_node[future]
            try:
                output = future.result()
                results.append(output)
            except Exception as exc:
                print(f"Node {node.name} generated an exception: {exc}")
                
    return results