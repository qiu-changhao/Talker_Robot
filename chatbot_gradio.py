import os
import re
from openai import OpenAI
import gradio as gr
import serpapi
# 初始化OpenAI客户端（阿里云百炼兼容接口）
def initialize_openai_client():
    return OpenAI(
        # 新加坡和北京地域的API Key不同
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 北京地域base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

# 使用Google Search API执行搜索
def search_google(query: str) -> str:
    """使用Google Search API执行搜索"""
    try:
        params = {
            "api_key": os.getenv("SERP_API_KEY"),
            "engine": "google",
            "q": query,
            "num": 3  # 获取3个搜索结果
        }

        # 使用serpapi.search函数执行搜索
        results = serpapi.search(params)

        # 处理搜索结果
        search_results = []
        search_results.append("### 搜索结果 ###")

        # 处理有机搜索结果
        if "organic_results" in results:
            for i, result in enumerate(results["organic_results"][:3], 1):
                title = result.get("title", "无标题")
                snippet = result.get("snippet", "无摘要")
                link = result.get("link", "无链接")
                search_results.append(f"\n**结果 {i}**:\n标题: {title}\n摘要: {snippet}\n链接: {link}")
        else:
            search_results.append("\n未找到相关搜索结果")

        return "\n".join(search_results)
    except Exception as e:
        return f"搜索错误: {str(e)}"

# 判断是否需要进行搜索
def should_search(query: str) -> bool:
    """判断用户是否需要搜索相关信息"""
    # 包含搜索相关关键词
    search_keywords = ["搜索", "查询", "最新", "现在", "今天", "最近", "目前", "多少", "多高", "多大", "现任"]
    for keyword in search_keywords:
        if keyword in query:
            return True
    
    # 包含具体的事实性问题
    question_patterns = [r"\?", r"是什么", r"为什么", r"怎么样", r"如何", r"何时", r"哪里", r"哪些"]
    for pattern in question_patterns:
        if re.search(pattern, query):
            return True
    
    return False

# 获取模型回复
def get_model_response(client, messages, use_search=True):
    """获取模型回复，如果需要则先进行搜索"""
    # 获取用户最后一条消息
    user_message = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), '')
    
    # 判断是否需要搜索
    if use_search and should_search(user_message):
        # 进行搜索
        search_results = search_google(user_message)
        
        # 将搜索结果添加到消息中
        search_messages = messages.copy()
        search_messages.append({"role": "system", "content": f"以下是相关的搜索信息：\n{search_results}\n请基于这些信息回答用户的问题。"})
        
        # 获取模型回复
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=search_messages
        )
        
        return completion.choices[0].message.content
    else:
        # 直接获取模型回复
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        return completion.choices[0].message.content
# Gradio聊天函数
def chat_with_bot(message, history, system_prompt, use_search=True):
    """Gradio聊天函数 - 使用messages格式"""
    # 初始化客户端
    client = initialize_openai_client()
    
    # 构建消息历史（包括用户自定义的system消息）
    system_message = {"role": "system", "content": system_prompt}
    
    # 添加历史消息
    messages = [system_message] + history.copy()
    
    # 添加当前用户消息
    user_message = {"role": "user", "content": message}
    messages.append(user_message)
    
    try:
        # 获取回复
        # 为了兼容get_model_response函数，我们需要提取content
        response_content = get_model_response(client, messages, use_search)
        
        # 创建助手回复消息
        assistant_message = {"role": "assistant", "content": response_content}
        
        # 返回更新后的历史记录
        return history + [user_message, assistant_message]
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        # 创建错误消息
        error_message = {"role": "assistant", "content": error_msg}
        return history + [user_message, error_message]

# 创建Gradio界面
def create_gradio_interface():
    with gr.Blocks(title="智能聊天助手") as demo:
        gr.Markdown("## 智能聊天助手")
        gr.Markdown("基于阿里云百炼和搜索功能的聊天机器人")
        
        # 自定义系统提示词输入框
        system_prompt = gr.Textbox(
            label="系统提示词",
            value="你是一个有帮助的助手，可以回答用户的问题。",
            placeholder="请输入系统提示词，定义AI助手的行为和能力"
        )
        
        # 控制选项 - 横向布局的顶部
        use_search = gr.Checkbox(label="启用联网搜索", value=True)
        
        # 聊天界面 - 占据下方整个区域
        chatbot = gr.Chatbot(label="聊天历史", type="messages", height=400)
        msg = gr.Textbox(label="输入消息")
        clear = gr.Button("清空对话")
        
        # 设置事件
        msg.submit(chat_with_bot, [msg, chatbot, system_prompt, use_search], chatbot)
        msg.submit(lambda: "", None, msg, queue=False)  # 确保用户输入后清空输入框
        clear.click(lambda: [], None, chatbot, queue=False)
        
    return demo

# 主函数
if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告: 环境变量 DASHSCOPE_API_KEY 未设置，请先设置阿里云百炼API密钥")
    
    if not os.getenv("SERP_API_KEY"):
        print("警告: 环境变量 SERP_API_KEY 未设置，搜索功能将不可用")
    
    # 创建并启动界面
    demo = create_gradio_interface()
    # 使用不同的端口以避免冲突
    demo.launch(server_name="127.0.0.1", server_port=7861)