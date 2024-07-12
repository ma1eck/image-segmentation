for i, node in enumerate(nodes):
        min_dist = float('inf')
        min_index = None
        for j, center in enumerate(k_means):
            distance = calculate_distance(node, center)
            if distance < min_dist:
                min_index = j
                min_dist = distance
        center_of_nodes[i] = min_index
        nodes_of_centers[min_index].append(node)