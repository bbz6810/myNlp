import re
from applications.knowledge_graph.query import Query


class QuestionTemplate:
    def __init__(self):
        self.question = None
        self.question_flag = None
        self.question_word = None

        self.q_template_dict = {
            0: self.get_movie_rating,
            1: self.get_movie_release_date,
            2: self.get_movie_type,
            3: self.get_movie_introduction,
            4: self.get_movie_actor_list,
            5: self.get_actor_info,
            6: self.get_actor_act_type_movie,
            7: self.get_actor_act_movie_list,
            8: self.get_movie_rating_bigger,
            9: self.get_movie_rating_smaller,
            10: self.get_actor_movie_type,
            11: self.get_cooperation_movie_list,
            12: self.get_actor_movie_num,
            13: self.get_actor_birthday
        }

        self.graph = Query()

    def get_question_answer(self, question, template_id_str):
        template_id, template_str = template_id_str.strip().split('\t')
        question_word, question_flag = [], []
        for one in question:
            question_word.append(one.split('/')[0])
            question_flag.append(one.split('/')[1])
        self.question = question
        self.question_word = question_word
        self.question_flag = question_flag
        answer = self.q_template_dict[int(template_id)]()
        return answer

    def get_name(self, type_str):
        name_count = self.question_flag.count(type_str)
        if name_count == 1:
            return self.question_word[self.question_flag.index(type_str)]
        else:
            return [self.question_word[idx] for idx, flag in enumerate(self.question_flag) if flag == type_str]

    def get_movie_name(self):
        return self.question_word[self.question_flag.index('nm')]

    def get_question_num_x(self):
        return re.sub('\D', '', ''.join(self.question_word))

    def get_movie_rating(self):  # 0 评分
        movie_name = self.get_movie_name()
        cql = f"match (m:Movie)-[]->() where m.title='{movie_name}' return m.rating"
        print('graph sql', cql)
        answer = self.graph.run(cql)[0]
        answer = round(answer, 2)
        final_answer = movie_name + "电影评分为" + str(answer) + "分！"
        return final_answer

    def get_movie_release_date(self):  # 1:nm 上映时间
        movie_name = self.get_movie_name()
        cql = f"match(m:Movie)-[]->() where m.title='{movie_name}' return m.releasedate"
        print('graph sql', cql)
        answer = self.graph.run(cql)[0]
        final_answer = movie_name + "的上映时间是" + str(answer) + "！"
        return final_answer

    def get_movie_type(self):  # 2:nm 类型
        movie_name = self.get_movie_name()
        cql = f"match(m:Movie)-[r:is]->(b) where m.title='{movie_name}' return b.name"
        print(cql)
        answer = self.graph.run(cql)
        answer = "、".join(list(set(answer)))
        final_answer = movie_name + "是" + str(answer) + "等类型的电影！"
        return final_answer

    def get_movie_introduction(self):  # 3:nm 简介
        movie_name = self.get_movie_name()
        cql = f"match(m:Movie)-[]->() where m.title='{movie_name}' return m.introduction"
        print(cql)
        answer = self.graph.run(cql)[0]
        final_answer = movie_name + "主要讲述了" + str(answer) + "！"
        return final_answer

    def get_movie_actor_list(self):  # 4:nm 演员列表
        movie_name = self.get_movie_name()
        cql = f"match(n:Person)-[r:actedin]->(m:Movie) where m.title='{movie_name}' return n.name"
        print(cql)
        answer = self.graph.run(cql)
        answer = "、".join(list(set(answer)))
        final_answer = movie_name + "由" + str(answer) + "等演员主演！"
        return final_answer

    def get_actor_info(self):  # 5:nnt 介绍
        actor_name = self.get_name('nr')
        cql = f"match(n:Person)-[]->() where n.name='{actor_name}' return n.biography"
        print(cql)
        answer = self.graph.run(cql)[0]
        return answer

    def get_actor_act_type_movie(self):  # 6:nnt ng 电影作品
        actor_name = self.get_name("nr")
        movie_type = self.get_name("ng")
        # 查询电影名称
        cql = f"match(n:Person)-[]->(m:Movie) where n.name='{actor_name}' return m.title"
        print(cql)
        movie_name_list = list(set(self.graph.run(cql)))
        # 查询类型
        result = []
        for movie_name in movie_name_list:
            movie_name = str(movie_name).strip()
            try:
                cql = f"match(m:Movie)-[r:is]->(t) where m.title='{movie_name}' return t.name"
                # print(cql)
                temp_type = self.graph.run(cql)
                if len(temp_type) == 0:
                    continue
                if movie_type in temp_type:
                    result.append(movie_name)
            except:
                pass
        answer = "、".join(['《{}》'.format(i) for i in result])
        print(answer)
        final_answer = actor_name + "演过的" + type + "电影有:\n" + answer + "。"
        return final_answer

    def get_actor_act_movie_list(self):  # 7:nnt 电影作品
        actor_name = self.get_name("nr")
        answer_list = self.get_actorname_movie_list(actor_name)
        answer = "、".join(['《{}》'.format(i) for i in answer_list])
        final_answer = actor_name + "演过" + str(answer) + "等电影！" if answer else '没有记录'
        return final_answer

    def get_actorname_movie_list(self, actorname):  # 查询电影名称

        cql = f"match(n:Person)-[]->(m:Movie) where n.name='{actorname}' return m.title"
        print(cql)
        answer = self.graph.run(cql)
        answer_list = list(set(answer))
        return answer_list

    def get_movie_rating_bigger(self):  # 8:nnt 参演评分 大于 x
        actor_name = self.get_name('nr')
        x = self.get_question_num_x()
        cql = f"match(n:Person)-[r:actedin]->(m:Movie) where n.name='{actor_name}' and m.rating>={x} return m.title"
        print(cql)
        answer = self.graph.run(cql)
        answer = "、".join(['《{}》'.format(i) for i in answer]).strip()
        final_answer = actor_name + "演的电影评分大于" + x + "分的有" + answer + "等！"
        return final_answer

    def get_movie_rating_smaller(self):  # 9:nnt 参演评分 小于 x
        actor_name = self.get_name('nr')
        x = self.get_question_num_x()
        cql = f"match(n:Person)-[r:actedin]->(m:Movie) where n.name='{actor_name}' and m.rating<{x} return m.title"
        print(cql)
        answer = self.graph.run(cql)
        answer = "、".join(['《{}》'.format(i) for i in answer]).strip()
        final_answer = actor_name + "演的电影评分小于" + x + "分的有" + answer + "等！"
        return final_answer

    def get_actor_movie_type(self):  # 10:nnt 演员参演电影的类型
        actor_name = self.get_name("nr")
        # 查询电影名称
        cql = f"match(n:Person)-[]->(m:Movie) where n.name='{actor_name}' return m.title"
        print(cql)
        movie_name_list = list(set(self.graph.run(cql)))
        # 查询类型
        result = []
        for movie_name in movie_name_list:
            movie_name = str(movie_name).strip()
            try:
                cql = f"match(m:Movie)-[r:is]->(t) where m.title='{movie_name}' return t.name"
                # print(cql)
                temp_type = self.graph.run(cql)
                if len(temp_type) == 0:
                    continue
                result += temp_type
            except:
                continue
        answer = "、".join(['【{}】'.format(i) for i in set(result)])
        print(answer)
        final_answer = actor_name + "演过的电影有" + answer + "等类型。"
        return final_answer

    def get_cooperation_movie_list(self):
        # 获取演员名字
        actor_name_list = self.get_name('nr')
        movie_list = {}
        for i, actor_name in enumerate(actor_name_list):
            answer_list = self.get_actorname_movie_list(actor_name)
            movie_list[i] = answer_list
        result_list = list(set(movie_list[0]).intersection(set(movie_list[1])))
        print(result_list)
        answer = "、".join(result_list)
        final_answer = actor_name_list[0] + "和" + actor_name_list[1] + "一起演过的电影主要是" + answer + "!"
        return final_answer

    def get_actor_movie_num(self):
        actor_name = self.get_name("nr")
        answer_list = self.get_actorname_movie_list(actor_name)
        movie_num = len(set(answer_list))
        answer = movie_num
        final_answer = actor_name + "演过" + str(answer) + "部电影!"
        return final_answer

    def get_actor_birthday(self):
        actor_name = self.get_name('nr')
        cql = f"match(n:Person)-[]->() where n.name='{actor_name}' return n.birth"
        print(cql)
        answer = self.graph.run(cql)[0]
        final_answer = actor_name + "的生日是" + answer + "。"
        return final_answer
