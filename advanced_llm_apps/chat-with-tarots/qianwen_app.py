from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from langchain_core.runnables import RunnableParallel # LCEL核心组件
from langchain_core.prompts import PromptTemplate
import streamlit as st
import helpers.qianwen_help_func as hf
import helpers.qwen_langchain_improved as qwen_improved
from PIL import Image
import os
from dotenv import load_dotenv



# --- Load the dataset ---
# 加载环境变量
try:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError("通义千问API密钥未配置，请确保.env文件中包含DASHSCOPE_API_KEY")
    
    # 使用改进版的通义千问LLM实现
    llm = qwen_improved.QwenLLMImproved(
        api_key=api_key,
        model="qwen-plus",  # 使用通义千问的plus模型
        temperature=0.8,
        top_p=0.8,
    )
    print(f"\n通义千问模型已配置成功")
except Exception as e:
    print(f"通义千问模型配置失败: {str(e)}")
    # 使用预定义的回退实现
    llm = hf.FallbackLLM(str(e))


csv_file_path = 'data/tarots.csv'
try:
    # Read CSV file
    df = pd.read_csv(csv_file_path, sep=';', encoding='latin1')
    print(f"CSV dataset loaded successfully: {csv_file_path}. Number of rows: {len(df)}")
    
    # Clean and normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Debug: Show column details
    print("\nDetails after cleanup:")
    for col in df.columns:
        print(f"Column: '{col}' (length: {len(col)})")
    
    # Define required columns (in lowercase)
    required_columns = ['card', 'upright', 'reversed', 'symbolism']
    
    # Verify all required columns are present
    available_columns = set(df.columns)
    missing_columns = [col for col in required_columns if col not in available_columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV file: {', '.join(missing_columns)}\n"
            f"Available columns: {', '.join(available_columns)}"
        )
    
    # Create card meanings dictionary with cleaned data
    card_meanings = {}
    for _, row in df.iterrows():
        card_name = row['card'].strip()
        card_meanings[card_name] = {
            'upright': str(row['upright']).strip() if pd.notna(row['upright']) else '',
            'reversed': str(row['reversed']).strip() if pd.notna(row['reversed']) else '',
            'symbolism': str(row['symbolism']).strip() if pd.notna(row['symbolism']) else ''
        }
    
    print(f"\nKnowledge base created with {len(card_meanings)} cards, meanings and symbolisms.")
    
except FileNotFoundError:
    print(f"Error: CSV File not found: {csv_file_path}")
    raise
except ValueError as e:
    print(f"Validation Error: {str(e)}")
    raise
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    raise



# --- Define the System Message Template ---
system_message_template = """
你是一位神秘的塔罗牌解读师，拥有深厚的象征学和心理学知识。
请基于提供的含义分析以下塔罗牌（同时考虑它们是否为逆位）：
{card_details}
请特别注意以下几个方面：
- 详细分析每张牌的含义（正位或逆位）。
- 然后根据卡牌提供一个与上下文相关的总体解释：{context}。
- 保持神秘，并基于特定列 {symbolism} 提供与卡牌象征意义相关的解释信息。
- 在解读结束时，总是提供改善或处理当前情况的建议。建议也请基于你的心理学知识。
"""

# --- Define the Prompt Template ---
prompt_analysis = PromptTemplate.from_template("{context}")

# --- Create the LangChain Chain ---
analyzer = (
    RunnableParallel(
        cards=lambda x: x['cards'],
        context=lambda x: x['context']
    )
    | (lambda x: hf.prepare_prompt_input(x, card_meanings))
    | (lambda x: [SystemMessage(content=system_message_template.format(**x)), HumanMessage(content=prompt_analysis.format(context=x['context']))])
    | llm
)

# --- Frontend Streamlit ---
st.set_page_config(
    page_title="🔮 交互式塔罗牌解读",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🔮 交互式塔罗牌解读")
st.markdown("欢迎来到您的个性化塔罗牌咨询！")
st.markdown("---")

num_cards = st.selectbox("🃏 选择您想要抽取的牌数（3张用于更聚焦的回答，7张用于更全面的概览）", [3, 5, 7])
context_question = st.text_area("✍️ 请在这里输入您的问题或背景。您可以用自然语言表达。", height=100)

if st.button("✨ 点亮您的道路：抽取并分析卡牌"):
    if not context_question:
        st.warning("为了获得更精确的解读，请输入您的问题或背景信息。")
    else:
        try:
            card_names_in_dataset = df['card'].unique().tolist()
            drawn_cards_list = hf.generate_random_draw(num_cards, card_names_in_dataset)
            st.subheader("✨ 您的卡牌已揭晓：")
            st.markdown("---")

            cols = st.columns(len(drawn_cards_list))
            for i, card_info in enumerate(drawn_cards_list):
                with cols[i]:
                    # The card_info['name'] from data/tarots.csv is now the direct image filename e.g., "00-thefool.jpg"
                    image_filename = card_info['name']
                    image_path = f"images/{image_filename}"
                    reversed_label = "(R)" if 'is_reversed' in card_info else ""
                    caption = f"{card_info['name']} {reversed_label}"

                    try:
                        img = Image.open(image_path)
                        if card_info.get('is_reversed', False):
                            img = img.rotate(180)
                        st.image(img, caption=caption, width=150)
                    except FileNotFoundError:
                        st.info(f"符号：{card_info['name']} {reversed_label} (在 {image_path} 未找到图片)")

            st.markdown("---")
            with st.spinner("🔮 正在揭示牌义..."):
                analysis_result = analyzer.invoke({"cards": drawn_cards_list, "context": context_question})
                st.subheader("📜 解读结果：")
                # 处理消息格式
                if hasattr(analysis_result, 'content'):
                    st.write(analysis_result.content)
                else:
                    st.write(str(analysis_result))

        except Exception as e:
            st.error(f"发生错误：{e}")

st.markdown("---")
st.info("请记住，卡牌提供洞察和反思；您的未来掌握在自己手中。")