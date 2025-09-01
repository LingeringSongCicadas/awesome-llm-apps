"""
改进版的通义千问LLM实现，更好地利用LangChain框架功能
"""

import os
import dashscope
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


from typing import Optional
from pydantic import Field

class QwenLLMImproved(BaseLanguageModel):
    """改进版的通义千问LLM实现"""
    
    # 使用Pydantic的Field来处理api_key，避免验证问题
    api_key: Optional[str] = Field(default=None, exclude=True)
    model: str = "qwen-plus"
    temperature: float = 0.8
    top_p: float = 0.8
    max_tokens: int = 2000
    
    def __init__(self, api_key=None, model="qwen-plus", temperature=0.8, top_p=0.8, max_tokens=2000):
        """
        初始化QwenLLMImproved实例
        
        参数:
            api_key: 通义千问API密钥
            model: 使用的模型名称
            temperature: 生成文本的随机性
            top_p: 采样概率阈值
            max_tokens: 最大生成token数
        """
        # 调用父类初始化方法
        super().__init__()
        
        # 优先使用传入的api_key，否则从环境变量获取
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("通义千问API密钥未配置，请提供api_key参数或设置DASHSCOPE_API_KEY环境变量")
        
        # 设置所有参数
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # 设置API密钥
        dashscope.api_key = self.api_key
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """生成文本响应"""
        try:
            # 将LangChain消息格式转换为通义千问API格式
            qwen_messages = self._convert_messages_to_qwen_format(messages)
            
            # 调用通义千问API
            response = dashscope.Generation.call(
                model=self.model,
                messages=qwen_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                repetition_penalty=kwargs.get('repetition_penalty', 1.0),
                stream=False,
                result_format='message'
            )
            
            # 处理API响应
            if response.status_code == 200 and 'output' in response and 'choices' in response['output']:
                content = response['output']['choices'][0]['message']['content']
                generations = [Generation(text=content)]
            else:
                error_msg = f"API调用失败: HTTP状态码 {response.status_code}, 错误信息: {response.get('message', '未知错误')}"
                generations = [Generation(text=error_msg)]
                print(error_msg)
        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            generations = [Generation(text=error_msg)]
            print(error_msg)
        
        return LLMResult(generations=[generations])
    
    def _convert_messages_to_qwen_format(self, messages):
        """将LangChain消息格式转换为通义千问API格式"""
        qwen_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                qwen_messages.append({'role': 'system', 'content': message.content})
            elif isinstance(message, HumanMessage):
                qwen_messages.append({'role': 'user', 'content': message.content})
            elif isinstance(message, AIMessage):
                qwen_messages.append({'role': 'assistant', 'content': message.content})
            else:
                # 默认作为用户消息处理
                qwen_messages.append({'role': 'user', 'content': str(message)})
        return qwen_messages
    
    def _llm_type(self):
        """返回LLM类型"""
        return "qwen"
    
    def invoke(self, input_data, config=None, stop=None, **kwargs):
        """调用LLM生成响应"""
        # 将输入转换为消息列表
        if isinstance(input_data, str):
            messages = [HumanMessage(content=input_data)]
        elif isinstance(input_data, list):
            messages = input_data
        else:
            messages = [HumanMessage(content=str(input_data))]
        
        # 生成响应
        result = self._generate(messages, stop=stop, **kwargs)
        
        # 返回第一个生成结果
        if result.generations and result.generations[0]:
            return AIMessage(content=result.generations[0][0].text)
        else:
            return AIMessage(content="生成响应失败")
    
    # 实现BaseLanguageModel所需的核心方法
    
    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """异步生成基于提示模板的响应"""
        # 使用同步版本作为基础实现
        return self.generate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)
        
    async def apredict(self, text, stop=None, callbacks=None, **kwargs):
        """异步预测单个文本的响应"""
        # 使用同步版本作为基础实现
        return self.predict(text, stop=stop, callbacks=callbacks, **kwargs)
        
    async def apredict_messages(self, messages, stop=None, callbacks=None, **kwargs):
        """异步预测消息列表的响应"""
        # 使用同步版本作为基础实现
        return self.predict_messages(messages, stop=stop, callbacks=callbacks, **kwargs)
    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """生成基于提示模板的响应"""
        # 简单实现，将提示转换为消息
        messages = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages.append([HumanMessage(content=prompt)])
            else:
                messages.append([HumanMessage(content=str(prompt))])
        
        # 调用已实现的_generate方法
        results = []
        for msg in messages:
            results.append(self._generate(msg, stop=stop, **kwargs))
        
        return results
    
    def predict(self, text, stop=None, callbacks=None, **kwargs):
        """预测单个文本的响应"""
        result = self._generate([HumanMessage(content=text)], stop=stop, **kwargs)
        if result.generations and result.generations[0]:
            return result.generations[0][0].text
        else:
            return "生成响应失败"
    
    def predict_messages(self, messages, stop=None, callbacks=None, **kwargs):
        """预测消息列表的响应"""
        result = self._generate(messages, stop=stop, **kwargs)
        if result.generations and result.generations[0]:
            return AIMessage(content=result.generations[0][0].text)
        else:
            return AIMessage(content="生成响应失败")