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



## 3、操作步骤
### （1） 环境配置
**<1> 新建虚拟环境** `conda create -n interpersonal_assistant python=3.10`

**<2> 激活虚拟环境** `conda activate interpersonal_assistant`

**<3> 安装所需的包** `pip install -r requirements.txt`

### （2） 运行应用程序
`streamlit run streamlit_app.py`

请到[智谱AI开放平台](https://open.bigmodel.cn/)申请API keys，填入页面左上角的框框后按回车键，即可开始问答。
