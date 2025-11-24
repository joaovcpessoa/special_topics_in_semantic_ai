import networkx as nx

# Criar um grafo direcionado
G = nx.DiGraph()

# Adicionar entidades com tipos
G.add_node("John", type="People")
G.add_node("Breno", type="People")
G.add_node("Python", type="Language")
G.add_node("Brazil", type="Country")

# Adicionar relações com rótulos
G.add_edge("John", "Breno", relation="friend_of")
G.add_edge("John", "Python", relation="program_in")
G.add_edge("Breno", "Brazil", relation="lives_in")

# Mostrar dados
print("Nodes:")
for node, data in G.nodes(data=True):
    print(f"  {node} -> {data}")

print("\rRelationships:")
for source, target, data in G.edges(data=True):
    print(f"  {source} --({data['relation']})--> {target}")
