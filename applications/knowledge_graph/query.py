from py2neo import Graph, Node, Relationship, NodeMatcher


class Query():
    def __init__(self):
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")

    # 问题类型0，查询电影得分
    def run(self, cql):
        # find_rela  = test_graph.run("match (n:Person{name:'张学友'})-[actedin]-(m:Movie) return m.title")
        result = []
        find_rela = self.graph.run(cql)
        for i in find_rela:
            result.append(i.items()[0][1])
        return result
