# 人际交往小助手
## 1、项目介绍

为应对大学生、初入职场的年轻人难以领会社交场中沟通要点的问题，开发了这样一款人际交往助手，使其更快融入和适应社会。

本项目使用LangChain调用ChatGLM，根据自己搜集整理的人际交往主题的本地知识文档，搭建向量知识库。使用搭建好的向量数据库，对query 查询问题进行召回，并将召回结果和query结合起来构建prompt，输入到大模型中进行问答。

目前知识库主要涵盖了人际交往中“自我介绍”、“开启话题”、“回应”、“请求”等几个主题的资料。

## 2、效果展示
### case1 

**问题：**

![1](https://github.com/adela778/interpersonal_communication_assistant/assets/154968495/641e1ab1-e04b-4d70-8dcc-ff75c9f370f8)


**原始chatglm的回答：**

![2](https://github.com/adela778/interpersonal_communication_assistant/assets/154968495/c6d63d04-756f-4311-a882-e981b78be542)


**本项目的回答：**

![3](https://github.com/adela778/interpersonal_communication_assistant/assets/154968495/c80efdb0-0263-4f05-bb88-a649b75e19fe)


### case2 

**问题：**

我是一名职员，在公司工作的时间不短了，也取得了一些亮眼的成绩，希望能得到加薪机会，请问我该如何向领导提出请求？具体该怎么说？

**原始chatglm的回答：**

向领导提出加薪请求时，你可以遵循以下步骤：

选择合适的时机：找一个工作环境相对轻松、领导心情较好的时候，这会让你的请求更容易被接受。

准备充分的理由：在提出加薪请求前，应该整理一份你在工作中的成绩和价值，例如：完成了哪些项目、为公司创造了哪些利润等。同时，你也可以通过调查同行业中类似职位的薪资水平，为自己的加薪请求提供支持。

表现出诚恳和自信：在提出请求时，用诚恳的语气表达你对目前工作的满意和对未来的期许，同时也要展示出自信，让领导看到你有能力胜任更高薪酬的工作。

强调你的价值和贡献：在谈话中，重点强调你对公司的价值和贡献，让领导了解到你的重要性。

表现出愿意协商的态度：提出加薪请求后，要做好心理准备，可能会遭遇拒绝或者需要协商。此时，要保持平和的心态，听取领导的意见，并表示自己愿意进行协商，以找到双方都能接受的结果。

你可以这样表达你的请求：

“领导，我想请您抽出几分钟时间，谈谈我的薪资问题。自从我加入公司以来，我一直在努力工作，并取得了一些成绩。在这段时间里，我完成了项目A、项目B，为公司创造了利润X。我在工作中得到了成长，也非常喜欢我们团队的工作氛围。我了解目前市场上与我职位类似的薪资水平，因此，我希望能得到一次加薪的机会，以便更好地体现我的价值。我知道公司有公司的考虑，如果我的要求过高，我也愿意进行协商。请您给我一个机会，让我们共同讨论一下我的薪资待遇。”

**本项目的回答：**

在向领导提出加薪请求时，可以采用“客观证明”和“STAR原则”来增加说服力。以下是一个可能的建议：

你可以说：

“领导，我最近回顾了一下我的工作成绩，发现在过去的六个月里，我负责的项目A和B都取得了超出预期的好成绩。比如，项目A通过我的数据整理和分析，我们优化了流程，提高了效率，为公司节省了成本，这个项目也被团队评价为最具创新性的案例之一。此外，项目B在我的带领下，我们团队提前完成了任务，并且获得了客户的高度评价，这对我们后续的合作有很大的帮助。”

（这里使用了STAR原则，即情境(Situation)、任务(Task)、行动(Action)、结果(Result)来详细描述你的工作成果。）

接着，可以这样继续：

“我也对比了一下行业内的薪酬水平，结合我目前的工作经验和业绩，我认为我有资格获得一定的薪酬调整。当然，我对公司的贡献不仅仅是这些，我也在不断学习，希望能为公司带来更多的价值。”

（这里你提出了客观的证据支持你的请求，并留给领导判断的空间，表现出尊重和理解。）

最后，可以这样结尾：

“无论结果如何，我都会继续努力工作，为公司的发展贡献自己的力量。”

（这样的表达方式既体现了你的自信和成绩，又显得客观、专业，同时也表现出对公司的忠诚和愿意继续努力的态度。）


## 3、操作步骤
### （1） 环境配置
**<1> 新建虚拟环境** `conda create -n interpersonal_assistant python=3.10`

**<2> 激活虚拟环境** `conda activate interpersonal_assistant`

**<3> 安装所需的包** `pip install -r requirements.txt`

### （2） 运行应用程序
`streamlit run streamlit_app.py`

请到[智谱AI开放平台](https://open.bigmodel.cn/)申请API keys，填入页面左上角的框框后按回车键，即可开始问答。
