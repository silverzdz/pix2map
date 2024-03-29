import numpy as np

def get_vector(centerlines, x, y):
    nodes = []
    lines = []
    for centerline in centerlines:
        line = []
        for point in centerline:
            if (point[0]-x <= -50) or (point[0]-x >= 50):
                continue
            if (point[1]-y <= -50) or (point[1]-y >= 50):
                continue
            node = (int("{:.0f}".format(point[0]-x)), int("{:.0f}".format(point[1]-y)))
            nodes.append(node)
            line.append(node)
        lines.append(line)
    return nodes, lines

def cal_adjacent_matrix(centerlines, x, y):
    # centerlines shape: [lines_num, 10, 3]
    nodes, lines = get_vector(centerlines, x, y)
    node_map = {}
    node_id = 0
    for node in nodes:
        if node in node_map.keys():
            continue
        node_map[node] = node_id
        node_id += 1
    node_num = node_id
    
    if len(node_map) > 512:
        node_map = {k: node_map[k] for k in list(node_map)[:512]}
    
    # adj_matrix = np.full((512, 512), -np.inf)
    
    ### use a big num instead of np.inf, for fear that the apprearance of nan after softmax
    adj_matrix = np.full((512, 512), -1e20)
    # adj_matrix = np.full((node_num, node_num), -np.inf)
    for line in lines:
        if len(line) == 0:
            continue
        line_length = len(line)
        
        first_node = line[0]
        last_node = line[-1]
        first_id = 0
        last_id = 0
        if first_node in node_map:
            first_id = node_map[first_node]
            adj_matrix[first_id][first_id] = 0
        if last_node in node_map:
            last_id = node_map[last_node]
            adj_matrix[last_id][last_id] = 0
        
        for i in range(1, line_length-1):
            node = line[i]
            if node in node_map:
                node_id = node_map[node]
                adj_matrix[node_id][node_id] = 0 # self
                if i == 1 and (first_node in node_map):
                    adj_matrix[first_id][node_id] = 0
                if i == line_length - 2 and (last_node in node_map):
                    adj_matrix[node_id][last_id] = 0
                '''
                if line[i-1] in node_map:
                    pre_id = node_map[line[i-1]]
                    adj_matrix[node_id][pre_id] = 0 # pre
                '''
                if line[i+1] in node_map:
                    next_id = node_map[line[i+1]]
                    adj_matrix[node_id][next_id] = 0 # next
    
    return node_map, adj_matrix  

def position_encoding(node_map):
    new_map = {}
    for i in node_map.keys():
        x = int(i[0]) + 50
        y = int(i[1]) + 50
        id = node_map[i]
        encoding = y * 100 + x
        encoding = 9999 if encoding >= 10000 else encoding
        new_map[encoding] = id
    return new_map