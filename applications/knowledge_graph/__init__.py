"""
    https://blog.csdn.net/xyz1584172808/article/details/89336478

    一、KG构建
    二、根据KG构建QA系统
        1、Q分类
        2、Q模板化
        3、根据模板返回答案

    整体框架：
        问题预处理：
            问题预处理->关键信息
            训练分类器->问题分类->问题模板

            问题意图->答案查询

    nr 名词-人名
    nm 名词-电影名
    ng 名词-类目


    question
    词性标注，名词替换
    问题分类，找出问题模板
    根据分词和问题模板作答

"""
