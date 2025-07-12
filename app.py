import streamlit as st
import google.generativeai as genai
import openai
import anthropic
import json
import time
from datetime import datetime
import re
from typing import Dict, List, Optional

# --- ページ設定 ---
st.set_page_config(
    page_title="物語創作 執筆支援ツール",
    page_icon="✍️",
    layout="wide"
)

# --- カスタムCSS ---
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.main-header h1 {
    color: white;
    text-align: center;
    margin: 0;
}
.quality-indicator {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.quality-high {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
}
.quality-medium {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
}
.quality-low {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
}
.writing-mode {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border: 2px solid #dee2e6;
    margin: 10px 0;
}
.ai-mode {
    background-color: #e3f2fd;
    border: 2px solid #2196f3;
}
.manual-mode {
    background-color: #fff3e0;
    border: 2px solid #ff9800;
}
.login-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 70vh;
}
.login-container h2 {
    margin-bottom: 1.5rem;
}
.glossary-sidebar .stTextInput > div > input {
    font-size: 0.9em;
}
.glossary-sidebar .stTextArea > div > textarea {
    font-size: 0.9em;
    height: 80px;
}
.st-emotion-grid {
    background-color: #f8f9fa;
}
/* 新規ユーザー登録・ログインフォーム用スタイル */
.auth-form {
    background-color: #f0f4f8;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #cce0f0;
    margin-top: 20px;
}
.auth-form h3 {
    color: #0056b3;
    margin-bottom: 15px;
}
.auth-form .stButton > button {
    width: 100%;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- ユーティリティ関数 ---

def count_tokens(text: str) -> int:
    """テキストのトークン数を推定（日本語対応）"""
    japanese_chars = len([c for c in text if ord(c) > 127])
    english_words = len(re.findall(r'\b\w+\b', text))
    estimated_tokens = int(japanese_chars * 1.5 + english_words * 1.3)
    if len(text) > 500:
        estimated_tokens += len(text) // 10
    return max(1, estimated_tokens)

def log_api_usage(prompt: str, response: str, model_name: str, prompt_tokens: int, response_tokens: int):
    """API使用量をログに記録"""
    total_tokens = prompt_tokens + response_tokens
    
    st.session_state.api_usage['daily_requests'] += 1
    st.session_state.api_usage['daily_tokens_used'] += total_tokens
    st.session_state.api_usage['total_requests'] += 1
    st.session_state.api_usage['total_tokens_used'] += total_tokens
    
    st.session_state.api_usage['request_history'].append({
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'prompt_tokens': prompt_tokens,
        'response_tokens': response_tokens,
        'total_tokens': total_tokens
    })
    
    if len(st.session_state.api_usage['request_history']) > 100:
        st.session_state.api_usage['request_history'] = st.session_state.api_usage['request_history'][-100:]

def get_model_name(provider: str) -> str:
    if provider == "Gemini":
        return "gemini-2.0-flash"
    elif provider == "OpenAI":
        return "gpt-4o-mini"
    elif provider == "Claude":
        return "claude-3-haiku-20240307"
    return ""

def analyze_synopsis_quality(synopsis: str) -> int:
    """あらすじの品質を簡易分析"""
    score = 0
    if 200 <= len(synopsis) <= 400: score += 30
    elif 150 <= len(synopsis) <= 500: score += 20
    else: score += 10
    
    engaging_words = ["しかし", "だが", "突然", "ついに", "果たして", "なぜなら", "そして", "もし", "驚くべきことに"]
    for word in engaging_words:
        if word in synopsis: score += 5
    
    if "?" in synopsis or "！" in synopsis: score += 10
    
    if any(keyword in synopsis for keyword in ["魔法", "異世界", "ドラゴン", "冒険"]): score += 15
    
    sentences = synopsis.split("。")
    sentences = [s.strip() for s in sentences if s.strip()]
    if 3 <= len(sentences) <= 6: score += 20
    
    return min(score, 100)

# --- マルチAIモデル API呼び出し関数 ---

def call_generative_api(prompt: str) -> Dict:
    """選択されたAIモデルのAPIを呼び出す統一関数"""
    model_provider = st.session_state.get('selected_model_provider', 'Gemini')
    # session_state からユーザー固有のAPIキーを取得
    api_keys = st.session_state.get('user_api_keys', {})
    
    model_name = get_model_name(model_provider)
    
    try:
        response_text = "エラー: 想定外のプロバイダーです。"
        prompt_tokens = 0
        response_tokens = 0

        if model_provider == "Gemini":
            api_key = api_keys.get('gemini')
            if not api_key: return {"text": "エラー: Gemini APIキーが設定されていません。", "prompt_tokens": 0, "response_tokens": 0}
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            prompt_tokens = count_tokens(prompt)
            
            response = model.generate_content(prompt)
            response_text = response.text
            response_tokens = count_tokens(response_text)

        elif model_provider == "OpenAI":
            api_key = api_keys.get('openai')
            if not api_key: return {"text": "エラー: OpenAI APIキーが設定されていません。", "prompt_tokens": 0, "response_tokens": 0}
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            response_tokens = response.usage.completion_tokens

        elif model_provider == "Claude":
            api_key = api_keys.get('claude')
            if not api_key: return {"text": "エラー: Anthropic (Claude) APIキーが設定されていません。", "prompt_tokens": 0, "response_tokens": 0}
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
            prompt_tokens = response.usage.input_tokens
            response_tokens = response.usage.output_tokens
        
        else:
            return {"text": "エラー: 不明なAIモデルが選択されています。", "prompt_tokens": 0, "response_tokens": 0}

        log_api_usage(prompt, response_text, model_name, prompt_tokens, response_tokens)
        
        st.session_state.current_call_token_info = {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens
        }

        return {"text": response_text, "prompt_tokens": prompt_tokens, "response_tokens": response_tokens}

    except Exception as e:
        error_message = f"API呼び出し中にエラーが発生しました: {str(e)}"
        st.session_state.current_call_token_info = {
            "model": model_name,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "error": str(e)
        }
        return {"text": error_message, "prompt_tokens": 0, "response_tokens": 0}

# --- AIコンテンツ生成関数 ---

def generate_ai_content(content_type: str, project_data: dict, additional_params: dict = None) -> str:
    """AI コンテンツ生成の統一関数"""
    base_info = f"""
作品基本情報:
- ジャンル: {project_data.get('genre', '未設定')}
- ターゲット読者: {project_data.get('target_audience', '未設定')}
- テーマ: {project.get('theme', '未設定')}
- あらすじ: {project.get('synopsis', '未設定')}
- 世界観: {project.get('world_setting', '未設定')[:500]}
"""
    prompt = ""
    if content_type == "synopsis":
        prompt = f"""
魅力的なライトノベルのあらすじを作成してください。

{base_info}
追加設定: {additional_params.get('custom_elements', '') if additional_params else ''}

要求:
1. 200-400文字の簡潔なあらすじ
2. 読者の興味を引く内容
3. 続きが気になる構成
4. 完成度の高い、魅力的な品質で作成してください。
"""
    elif content_type == "character":
        prompt = f"""
魅力的なライトノベルのキャラクターを作成してください。

{base_info}
キャラクター要求:
- 名前: {additional_params.get('char_name', '') if additional_params else ''}
- 役割: {additional_params.get('char_role', '') if additional_params else ''}
- 追加要求: {additional_params.get('char_details', '') if additional_params else ''}

作成項目:
1. 詳細な性格設定
2. 背景・過去
3. 目標・動機
4. 外見・特徴
5. 口調・話し方
6. 他キャラとの関係性

読者に愛される魅力的なキャラクターを設計してください。
"""
    elif content_type == "world_setting":
        prompt = f"""
独創的で魅力的な世界観を構築してください。

{base_info}
世界観要求: {additional_params.get('world_elements', '') if additional_params else ''}

構築項目:
1. 世界の基本ルール・法則
2. 歴史・背景
3. 政治・社会システム
4. 魔法・超能力システム（該当する場合）
5. 地理・環境
6. 文化・風習
7. 技術レベル

既存作品との差別化を意識した独創的な世界観を作成してください。
"""
    elif content_type == "chapter":
        char_info = ""
        if project_data.get('characters'):
            char_list_display = list(project_data['characters'].keys())[:5]
            char_info = f"\n主要キャラクター（抜粋）:\n{', '.join(char_list_display)}"
        
        prompt = f"""
読者を引き込む魅力的な章を執筆してください。

{base_info}{char_info}
章の設定:
- チャプター名/番号: {additional_params.get('chapter_name', '第X章') if additional_params else '第X章'}
- プロット概要: {additional_params.get('chapter_plot', '指定なし') if additional_params else '指定なし'}
- 文字数目標: {additional_params.get('target_length', '3000-5000') if additional_params else '3000-5000'}文字
- 文体: {additional_params.get('writing_style', '三人称') if additional_params else '三人称'}

執筆要求:
1. 魅力的な導入
2. キャラクターの魅力を最大化
3. 読者を飽きさせない展開
4. 次章への引き
5. 完成度の高い文章力

多くの読者に楽しんでもらえる品質で執筆してください。
"""
    elif content_type == "full_story":
        prompt = f"""
完全なライトノベル作品を執筆してください。

{base_info}
執筆要求:
- 文字数: {additional_params.get('target_length', '10000-15000') if additional_params else '10000-15000'}文字
- 章数: {additional_params.get('chapter_count', '3-5') if additional_params else '3-5'}章構成
- 文体: {additional_params.get('writing_style', '三人称') if additional_params else '三人称'}

構成:
1. 魅力的なプロローグ
2. キャラクター紹介と世界観提示
3. 事件・問題の発生
4. 展開・クライマックス
5. 解決・エピローグ

素晴らしい品質で作成してください。各章の終わりに【第○章 終了】と明記してください。
"""

    api_response = call_generative_api(prompt)
    return api_response['text']

def modify_content_with_ai(content: str, modification_request: str, content_type: str = "テキスト") -> str:
    """AIを使ってコンテンツを修正する"""
    modification_prompt = f"""
以下の{content_type}を、ユーザーの指示に従って修正してください。

【修正指示】
{modification_request}

【現在の{content_type}】
{content}

【修正要求】
- 修正指示に沿って内容を改善してください。
- 元のテキストの良い点は維持しつつ、指示された変更を加えてください。
- {content_type}として自然で読みやすい文章にしてください。
- 以下の点は必ず守ってください：
    - {content_type}の意図や魅力を損なわないこと。
    - 文体が不自然にならないように注意してください。

修正された{content_type}のみを出力してください。余計な説明は不要です。
"""
    api_response = call_generative_api(modification_prompt)
    return api_response['text']

# --- セッションステート初期化 ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None # ログインしているユーザー名を保持
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
# APIキーはユーザーごとに管理するため、session_state.user_api_keys を使用
if 'user_api_keys' not in st.session_state:
    st.session_state.user_api_keys = {} # { 'username': {'gemini': '...', 'openai': '...'}, ... }
if 'selected_model_provider' not in st.session_state:
    st.session_state.selected_model_provider = "Gemini"
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = {
        'daily_requests': 0, 'daily_tokens_used': 0,
        'last_reset_date': datetime.now().date().isoformat(),
        'total_requests': 0, 'total_tokens_used': 0,
        'request_history': []
    }
if 'current_call_token_info' not in st.session_state:
    st.session_state.current_call_token_info = {}

# プロジェクトデータ構造に glossary を追加
if 'projects' in st.session_state:
    for project_name, project_data in st.session_state.projects.items():
        if 'glossary' not in project_data:
            project_data['glossary'] = {}

# 日付リセット
current_date = datetime.now().date().isoformat()
if st.session_state.api_usage['last_reset_date'] != current_date:
    st.session_state.api_usage.update({'daily_requests': 0, 'daily_tokens_used': 0, 'last_reset_date': current_date})

# --- 認証処理（初回ユーザー設定） ---

def setup_user_view():
    """初めてアプリを使うユーザー向けの、ユーザー名とパスワード設定画面"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/ios-filled/50/000000/book.png", width=100)
    st.title("物語創作 執筆支援ツール")
    st.subheader("ようこそ！")
    
    st.markdown("アカウントを作成し、創作を開始しましょう。", unsafe_allow_html=True)
    
    # st.form を使用する箇所に submit ボタンを追加し、st.h3 を修正
    with st.form("user_setup_form", clear_on_submit=True):
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        # st.h3("アカウント設定") -> st.markdown("### アカウント設定") に修正
        st.markdown("### アカウント設定", unsafe_allow_html=True) # 見出しの修正
        
        new_username = st.text_input("希望するユーザー名")
        new_password = st.text_input("パスワード設定", type="password")
        confirm_password = st.text_input("パスワード確認", type="password")
        
        # submit ボタンを追加
        submitted = st.form_submit_button("アカウントを作成して開始")
        
        if submitted: # submit ボタンが押された場合のみ以下の処理を実行
            if new_username and new_password and confirm_password:
                if new_password == confirm_password:
                    st.session_state.registered_username = new_username
                    st.session_state.registered_password = new_password
                    st.session_state.current_user = new_username
                    st.session_state.logged_in = True
                    st.session_state.user_api_keys[new_username] = {}
                    save_user_data()
                    st.success(f"アカウント「{new_username}」が作成されました！")
                    st.rerun()
                else:
                    st.error("パスワードが一致しません。")
            else:
                st.error("ユーザー名とパスワードを両方入力してください。")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def login_view():
    """ログイン画面"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/ios-filled/50/000000/book.png", width=100)
    st.title("物語創作 執筆支援ツール")
    st.subheader("ログイン")
    
    with st.form("login_form", clear_on_submit=True):
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        st.h3("ログイン")
        
        login_username = st.text_input("ユーザー名", key="login_username_input")
        login_password = st.text_input("パスワード", type="password", key="login_password_input")
        
        login_button = st.form_submit_button("ログイン")
        
        if login_button:
            if authenticate_user(login_username, login_password):
                st.success(f"ようこそ、{st.session_state.current_user}さん！")
                st.rerun()
            else:
                st.error("ユーザー名またはパスワードが間違っています。")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 用語集管理のサイドバー表示関数 ---
def glossary_sidebar_view():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 用語集管理")
    
    if not st.session_state.current_user: # ログインしていない場合は表示しない
        st.sidebar.info("ログインしてください。")
        return

    if not st.session_state.current_project:
        st.sidebar.info("プロジェクトを選択してください。")
        return

    project = st.session_state.projects[st.session_state.current_project]
    if 'glossary' not in project:
        project['glossary'] = {}
    glossary = project['glossary']

    with st.sidebar.expander("用語集へ追加", expanded=False):
        new_term_name = st.text_input("用語名", key="glossary_term_name_input")
        new_term_description = st.text_area("説明", key="glossary_term_description_input")
        
        if st.button("追加", key="add_glossary_term_btn"):
            if new_term_name and new_term_description:
                if new_term_name not in glossary:
                    glossary[new_term_name] = {
                        'description': new_term_description,
                        'added_at': datetime.now().isoformat()
                    }
                    st.sidebar.success(f"「{new_term_name}」を用語集に追加しました。")
                    st.rerun()
                else:
                    st.sidebar.warning(f"「{new_term_name}」は既に用語集に存在します。")
            else:
                st.sidebar.warning("用語名と説明の両方を入力してください。")

    st.sidebar.markdown("---")
    st.sidebar.subheader("登録済み用語")
    
    if not glossary:
        st.sidebar.info("用語集はまだ登録されていません。")
    else:
        search_term = st.sidebar.text_input("用語を検索", key="glossary_search_input", placeholder="例：アルカナライト")
        
        filtered_glossary_keys = [term for term in glossary if search_term.lower() in term.lower()]
        
        if not filtered_glossary_keys:
            st.sidebar.warning("該当する用語は見つかりませんでした。")
        else:
            for term_name in sorted(filtered_glossary_keys):
                term_data = glossary[term_name]
                with st.sidebar.expander(f"📚 {term_name}", expanded=False):
                    st.write(f"**説明:** {term_data.get('description', '未設定')}")
                    
                    col_term_edit, col_term_delete = st.columns(2)
                    with col_term_edit:
                        if st.button("編集", key=f"edit_glossary_{term_name}"):
                            st.session_state.editing_glossary_term = term_name
                            st.rerun()
                    with col_term_delete:
                        if st.button("削除", key=f"delete_glossary_{term_name}"):
                            if st.sidebar.button(f"確定: '{term_name}' を削除", key=f"confirm_delete_glossary_{term_name}"):
                                del glossary[term_name]
                                st.sidebar.success(f"「{term_name}」を削除しました。")
                                st.rerun()

    if 'editing_glossary_term' in st.session_state and st.session_state.editing_glossary_term:
        term_to_edit = st.session_state.editing_glossary_term
        term_data_orig = glossary.get(term_to_edit)
        
        if term_data_orig:
            with st.dialog(f"「{term_to_edit}」を編集", key="edit_glossary_dialog"):
                edited_term_name = st.text_input("用語名", value=term_to_edit, key=f"edit_glossary_name_input_{term_to_edit}")
                edited_term_description = st.text_area("説明", value=term_data_orig.get('description', ''), key=f"edit_glossary_description_input_{term_to_edit}", height=120)
                
                col_edit_save, col_edit_cancel = st.columns(2)
                with col_edit_save:
                    if st.button("保存", key=f"save_glossary_edit_{term_to_edit}"):
                        if edited_term_name and edited_term_description:
                            if edited_term_name != term_to_edit and edited_term_name in glossary:
                                st.error(f"「{edited_term_name}」は既に用語集に存在します。")
                            else:
                                if edited_term_name != term_to_edit:
                                    del glossary[term_to_edit]
                                
                                glossary[edited_term_name] = {
                                    'description': edited_term_description,
                                    'added_at': term_data_orig.get('added_at', datetime.now().isoformat())
                                }
                                del st.session_state.editing_glossary_term
                                st.success(f"用語「{edited_term_name}」を更新しました。")
                                st.rerun()
                        else:
                            st.warning("用語名と説明を入力してください。")
                with col_edit_cancel:
                    if st.button("キャンセル", key=f"cancel_glossary_edit_{term_to_edit}"):
                        del st.session_state.editing_glossary_term
                        st.rerun()

# --- メインコンテンツ表示関数 ---
def main_app_view():
    # --- サイドバー ---
    st.sidebar.title("🔧 設定")

    # APIキー設定セクション
    st.sidebar.subheader("🧠 AIモデル設定")
    st.session_state.selected_model_provider = st.sidebar.selectbox(
        "使用するAIモデル",
        ["Gemini", "OpenAI", "Claude"],
        index=["Gemini", "OpenAI", "Claude"].index(st.session_state.selected_model_provider)
    )

    # ユーザー固有のAPIキー設定フィールド
    st.sidebar.subheader("🔑 APIキー設定")
    current_user = st.session_state.current_user
    user_api_keys = st.session_state.user_api_keys.get(current_user, {})

    # サイドバーの入力フィールドに現在の値を反映させるために session_state を使う
    # secrets.tomlからの値は、初回起動時や初回ログイン時に session_state に初期値として設定する方が良い
    # ここでは、既に session_state.user_api_keys に保存されている値を入力フィールドに表示する
    
    gemini_key_input = st.sidebar.text_input(
        "Google Gemini API Key", 
        type="password", 
        value=user_api_keys.get('gemini', ''), 
        key=f"user_gemini_api_key_input_{current_user}",
        help="Gemini 2.0 Flash を使う場合もここに入力します。"
    )
    openai_key_input = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password", 
        value=user_api_keys.get('openai', ''), 
        key=f"user_openai_api_key_input_{current_user}"
    )
    claude_key_input = st.sidebar.text_input(
        "Anthropic (Claude) API Key", 
        type="password", 
        value=user_api_keys.get('claude', ''), 
        key=f"user_claude_api_key_input_{current_user}"
    )
    
    # 入力されたAPIキーを session_state.user_api_keys に保存
    if gemini_key_input != user_api_keys.get('gemini'):
        user_api_keys['gemini'] = gemini_key_input
    if openai_key_input != user_api_keys.get('openai'):
        user_api_keys['openai'] = openai_key_input
    if claude_key_input != user_api_keys.get('claude'):
        user_api_keys['claude'] = claude_key_input

    st.session_state.user_api_keys[current_user] = user_api_keys # 更新したAPIキーをセッションステートに保存


    def is_api_key_set():
        provider = st.session_state.selected_model_provider.lower()
        # 現在ログインしているユーザーのAPIキーを使用
        return bool(st.session_state.user_api_keys.get(current_user, {}).get(provider))

    # API使用状況表示
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 API使用状況")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("今日のリクエスト", st.session_state.api_usage['daily_requests'])
    col2.metric("今日のトークン", f"{st.session_state.api_usage['daily_tokens_used']:,}")
    st.sidebar.metric("総リクエスト数", st.session_state.api_usage['total_requests'])
    st.sidebar.metric("総トークン数", f"{st.session_state.api_usage['total_tokens_used']:,}")
    
    if st.session_state.current_call_token_info:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 直近のAPI呼び出し")
        token_info = st.session_state.current_call_token_info
        st.sidebar.write(f"**モデル:** {token_info.get('model', 'N/A')}")
        if 'error' in token_info:
            st.sidebar.error(f"**エラー:** {token_info['error'][:50]}...")
        else:
            st.sidebar.write(f"**プロンプト:** {token_info.get('prompt_tokens', 0):,} トークン")
            st.sidebar.write(f"**レスポンス:** {token_info.get('response_tokens', 0):,} トークン")
            st.sidebar.write(f"**合計:** {token_info.get('total_tokens', 0):,} トークン")

    # 用語集管理サイドバーを表示
    glossary_sidebar_view()

    # --- メインヘッダー ---
    st.markdown("""
    <div class="main-header">
        <h1>✍️ 物語創作 執筆支援ツール</h1>
        <p style="text-align: center; color: white; margin: 0;">あなたの創作活動を、アイデア出しから完成までサポートします。</p>
    </div>
    """, unsafe_allow_html=True)

    # --- プロジェクト管理 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 プロジェクト管理")

    with st.sidebar.expander("新規プロジェクト作成"):
        new_project_name = st.text_input("プロジェクト名")
        if st.button("作成") and new_project_name:
            if new_project_name not in st.session_state.projects:
                st.session_state.projects[new_project_name] = {
                    'created_at': datetime.now().isoformat(), 'synopsis': '', 'characters': {},
                    'world_setting': '', 'plot_outline': '', 'chapters': {}, 'genre': '',
                    'target_audience': '', 'theme': '', 'writing_mode': 'manual',
                    'glossary': {}
                }
                st.session_state.current_project = new_project_name
                st.success(f"プロジェクト「{new_project_name}」を作成しました。")
                st.rerun()
            else:
                st.warning(f"プロジェクト「{new_project_name}」は既に存在します。")

    if st.session_state.projects:
        project_keys = list(st.session_state.projects.keys())
        if st.session_state.current_project not in project_keys:
            st.session_state.current_project = project_keys[0] if project_keys else None
            
        current_project_index = project_keys.index(st.session_state.current_project) if st.session_state.current_project in project_keys else 0
        
        selected_project = st.sidebar.selectbox("現在のプロジェクト", project_keys, index=current_project_index)
        
        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            st.rerun()

        if st.button("🗑️ 現在のプロジェクトを削除"):
            if st.session_state.current_project:
                del st.session_state.projects[st.session_state.current_project]
                st.session_state.current_project = None
                st.success("プロジェクトを削除しました。")
                st.rerun()

    # データ管理
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 データ管理")
    if st.session_state.projects:
        project_json_all = json.dumps(st.session_state.projects, ensure_ascii=False, indent=2)
        st.sidebar.download_button(
            label="📤 全プロジェクトをエクスポート",
            data=project_json_all,
            file_name="novel_projects_all.json",
            mime="application/json"
        )

    uploaded_file = st.sidebar.file_uploader("📥 プロジェクトをインポート", type="json")
    if uploaded_file is not None:
        if st.sidebar.button("インポート実行"):
            try:
                imported_data = json.load(uploaded_file)
                st.session_state.projects.update(imported_data)
                for project_name, project_data in st.session_state.projects.items():
                    if 'glossary' not in project_data:
                        project_data['glossary'] = {}
                st.sidebar.success("インポート完了！")
                st.rerun()
            except json.JSONDecodeError:
                st.sidebar.error("無効なJSONファイルです。")
            except Exception as e:
                st.sidebar.error(f"インポートエラー: {e}")

    # --- メインコンテンツ ---
    if st.session_state.current_project:
        project = st.session_state.projects[st.session_state.current_project]
        
        st.subheader("✍️ 執筆モード選択")
        current_writing_mode = project.get('writing_mode', 'manual')
        
        col_radio1, col_radio2, col_radio3 = st.columns(3)
        with col_radio1:
            if st.button("🖊️ セルフ執筆", help="自分で執筆します。AIはアイデア出しや推敲に使います。", use_container_width=True):
                project['writing_mode'] = 'manual'
                st.rerun()
        with col_radio2:
            if st.button("🤖 AI執筆支援", help="AIに生成してもらい、それを基にあなたの創作を広げます。", use_container_width=True):
                project['writing_mode'] = 'ai'
                st.rerun()
        with col_radio3:
            if st.button("🔄 ハイブリッド", help="手動とAI生成を組み合わせて効率的に進めます。", use_container_width=True):
                project['writing_mode'] = 'hybrid'
                st.rerun()
        
        mode_class = ""
        if current_writing_mode == 'manual':
            mode_class = "manual-mode"
            mode_text = "🖊️ セルフ執筆モード"
        elif current_writing_mode == 'ai':
            mode_class = "ai-mode"
            mode_text = "🤖 AI執筆支援モード"
        else: # hybrid
            mode_class = ""
            mode_text = "🔄 ハイブリッドモード"
            
        st.markdown(f'<div class="writing-mode {mode_class}"><strong>{mode_text}</strong></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 企画・設定", "👥 キャラクター", "🗺️ 世界観", "📖 執筆", "🔍 品質チェック", "📊 分析・改善"])
        
        with tab1: # 企画・設定
            st.header("📋 作品企画・基本設定")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("基本情報")
                if current_writing_mode == 'manual' or current_writing_mode == 'hybrid':
                    all_genres = ["異世界ファンタジー", "現代ファンタジー", "学園もの", "SF", "ミステリー", "恋愛", "バトル・アクション", "日常系", "ホラー・サスペンス", "その他"]
                    genre_index = all_genres.index(project.get('genre', '')) if project.get('genre') in all_genres else 0
                    project['genre'] = st.selectbox("ジャンル（メイン）", all_genres, index=genre_index)
                    
                    all_targets = ["中高生男性", "中高生女性", "大学生・20代男性", "大学生・20代女性", "30代以上", "全年齢", "特定ターゲット"]
                    target_index = all_targets.index(project.get('target_audience', '')) if project.get('target_audience') in all_targets else 0
                    project['target_audience'] = st.selectbox("ターゲット読者層", all_targets, index=target_index)

                    project['theme'] = st.text_input("作品テーマ（核となるメッセージ）", value=project.get('theme', ''), placeholder="例：友情の大切さ、成長と自立、愛と犠牲...")

                if current_writing_mode == 'ai' or current_writing_mode == 'hybrid':
                    st.subheader("🤖 AI自動生成設定")
                    genre_preference = st.selectbox("好みのジャンル", ["おまかせ", "異世界", "学園", "SF", "恋愛", "バトル", "ファンタジー", "ミステリー"], key="ai_genre_pref")
                    target_preference = st.selectbox("ターゲット読者", ["おまかせ", "男性向け", "女性向け", "全年齢"], key="ai_target_pref")
                    tone_preference = st.selectbox("作品の雰囲気", ["おまかせ", "明るい", "シリアス", "コメディ", "ダーク", "感動的", "サスペンスフル"], key="ai_tone_pref")
                    
                    if st.button("🎯 AI企画生成"):
                        if not is_api_key_set():
                            st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                        else:
                            with st.spinner("企画生成中..."):
                                genre_map = {"異世界": "異世界ファンタジー", "学園": "学園もの", "SF": "SF", "恋愛": "恋愛", "バトル": "バトル・アクション", "ファンタジー": "現代ファンタジー", "ミステリー": "ミステリー", "おまかせ": "異世界ファンタジー"}
                                target_map = {"男性向け": "中高生男性", "女性向け": "中高生女性", "全年齢": "全年齢", "おまかせ": "中高生男性"}
                                project['genre'] = genre_map.get(genre_preference, "その他")
                                project['target_audience'] = target_map.get(target_preference, "特定ターゲット")
                                
                                theme_prompt = f"ジャンル「{project['genre']}」、読者層「{project['target_audience']}」、雰囲気「{tone_preference}」の物語に適した、ライトノベルの読者が興味を惹かれるような魅力的なテーマを1つ、15文字以内で簡潔に提案してください。"
                                
                                api_response = call_generative_api(theme_prompt)
                                if not api_response['text'].startswith("エラー"):
                                    project['theme'] = api_response['text'].strip()
                                    st.success("企画を自動生成しました！")
                                    st.rerun()
                                else:
                                    st.error(api_response['text'])
                                    project['theme'] = "成長と友情の物語"
                            
            with col2:
                st.subheader("あらすじ・コンセプト")
                if current_writing_mode == 'manual' or current_writing_mode == 'hybrid':
                    project['synopsis'] = st.text_area("作品あらすじ（200-400文字）", value=project.get('synopsis', ''), height=150, help="読者が最初に見る重要な要素。魅力的で続きが気になる内容に")

                if current_writing_mode == 'ai' or current_writing_mode == 'hybrid':
                    st.subheader("🤖 AI あらすじ生成")
                    custom_elements = st.text_area("追加要望（オプション）", placeholder="例：主人公は料理が得意、ドラゴンが登場、切ないラブコメ要素...", height=80, key="synopsis_custom_elements")
                    if st.button("✨ AIあらすじ生成", key="generate_synopsis_btn"):
                        if not is_api_key_set():
                            st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                        else:
                            with st.spinner("あらすじ生成中..."):
                                ai_synopsis = generate_ai_content("synopsis", project, {"custom_elements": custom_elements})
                                if not ai_synopsis.startswith("エラー"):
                                    project['synopsis'] = ai_synopsis
                                    st.success("あらすじを生成しました！")
                                    st.rerun()
                                else:
                                    st.error(ai_synopsis)

                if project.get('synopsis'):
                    with st.expander("🔧 あらすじ修正 (AI)"):
                        synopsis_modification = st.text_area("修正指示", placeholder="例：もっと感動的に、謎めいた要素を追加、主人公の心情を丁寧に...", height=60, key="synopsis_mod")
                        if st.button("🤖 あらすじを修正", key="modify_synopsis_btn") and synopsis_modification:
                            if not is_api_key_set():
                                st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                            else:
                                with st.spinner("修正中..."):
                                    modified_synopsis = modify_content_with_ai(project['synopsis'], synopsis_modification, "あらすじ")
                                    if not modified_synopsis.startswith("エラー"):
                                        st.session_state.modified_synopsis = modified_synopsis
                                        st.success("修正案が作成されました。内容を確認し、採用ボタンを押してください。")
                                        st.rerun()
                                    else:
                                        st.error(modified_synopsis)
                    
                    if 'modified_synopsis' in st.session_state and st.session_state.modified_synopsis:
                        st.write("#### 修正案の確認")
                        col_rev1, col_rev2 = st.columns(2)
                        with col_rev1:
                            st.write("**修正前**"); st.write(project['synopsis'])
                        with col_rev2:
                            st.write("**修正後**"); st.write(st.session_state.modified_synopsis)
                        
                        if st.button("✅ 修正版を採用", key="accept_synopsis_mod"):
                            project['synopsis'] = st.session_state.modified_synopsis
                            del st.session_state.modified_synopsis
                            st.success("修正版を採用しました！")
                            st.rerun()

                    synopsis_score = analyze_synopsis_quality(project['synopsis'])
                    if synopsis_score >= 80: st.markdown('<div class="quality-indicator quality-high">✅ あらすじ品質: 高</div>', unsafe_allow_html=True)
                    elif synopsis_score >= 60: st.markdown('<div class="quality-indicator quality-medium">⚠️ あらすじ品質: 中（改善推奨）</div>', unsafe_allow_html=True)
                    else: st.markdown('<div class="quality-indicator quality-low">❌ あらすじ品質: 低（要改善）</div>', unsafe_allow_html=True)

        with tab2: # キャラクター
            st.header("👥 キャラクター設定")
            st.subheader("既存キャラクター")
            if not project.get('characters'):
                st.info("まだキャラクターは登録されていません。")
            else:
                for name, data in project['characters'].items():
                    with st.expander(f"👤 {name} ({data.get('role', '役割不明')})"):
                        st.write(f"**役割:** {data.get('role', '未設定')}")
                        st.write(f"**性格:** {data.get('personality', '未設定')}")
                        st.write(f"**背景:** {data.get('background', '未設定')}")
                        st.write(f"**外見:** {data.get('appearance', '未設定')}")
                        st.write(f"**口調:** {data.get('speech', '未設定')}")
                        if st.button(f"{name} の詳細をAIで編集", key=f"edit_char_{name}"):
                            st.session_state.editing_character = name
                            st.rerun()

            st.subheader("新キャラクター作成")
            col_char_name, col_char_role, col_char_mode = st.columns([2, 2, 1])
            with col_char_name: new_char_name = st.text_input("キャラクター名", key="new_char_name_input")
            with col_char_role: new_char_role = st.selectbox("役割", ["主人公", "ヒロイン", "ライバル", "親友", "師匠", "敵役", "サポート", "その他"], key="new_char_role_select")
            with col_char_mode: char_creation_mode = st.radio("作成方法", ["✋ 手動", "🤖 AI"], key="char_creation_mode_radio")

            char_details_input = ""
            if char_creation_mode == "🤖 AI":
                char_details_input = st.text_area("キャラクター詳細要望（AI生成用）", placeholder="例：クールで無口、実は情深い、剣術が得意、過去に因縁あり...", height=80, key="char_ai_details")
            
            if st.button("➕ キャラクターを追加", key="add_character_btn"):
                if new_char_name and (char_creation_mode == "✋ 手動" or char_details_input):
                    if new_char_name not in project['characters']:
                        char_data = {'role': new_char_role}
                        if char_creation_mode == "✋ 手動":
                            char_data['details'] = "手動入力用の詳細欄を追加してください。"
                        elif char_creation_mode == "🤖 AI":
                            if not is_api_key_set():
                                st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                            else:
                                with st.spinner("キャラクター生成中..."):
                                    full_char_prompt = f"""
以下の情報を基に、ライトノベルのキャラクター設定を詳細に生成してください。
キャラクター名: {new_char_name}
役割: {new_char_role}
要望: {char_details_input}

生成項目:
1. 詳細な性格設定（長所、短所、癖など）
2. 背景・過去（物語に影響を与える要素）
3. 目標・動機
4. 外見的特徴（髪の色、目の色、体格、服装など）
5. 口調・話し方
6. 他キャラクターとの関係性（想定されるもの）
7. そのキャラクターを表す象徴的なアイテムや能力（あれば）

読者に愛されるような、深みのあるキャラクター設定を作成してください。
"""
                                    api_response = call_generative_api(full_char_prompt)
                                    if not api_response['text'].startswith("エラー"):
                                        char_data['details'] = api_response['text']
                                        st.success(f"キャラクター「{new_char_name}」をAIで生成しました！")
                                        st.rerun()
                                    else:
                                        st.error(api_response['text'])
                                        char_data['details'] = "AI生成に失敗しました。"
                        
                        project['characters'][new_char_name] = char_data
                        st.success(f"キャラクター「{new_char_name}」を追加しました。")
                        st.rerun()
                    else:
                        st.warning(f"キャラクター「{new_char_name}」は既に存在します。")
                else:
                    st.warning("キャラクター名と、手動入力またはAI生成のための情報が必要です。")

            if 'editing_character' in st.session_state and st.session_state.editing_character:
                char_to_edit = st.session_state.editing_character
                char_data_orig = project['characters'][char_to_edit]
                
                with st.dialog(f"{char_to_edit} の詳細を編集", key="edit_char_dialog"):
                    edited_char_name = st.text_input("キャラクター名", value=char_to_edit, key=f"edit_name_{char_to_edit}")
                    edited_char_role = st.selectbox("役割", ["主人公", "ヒロイン", "ライバル", "親友", "師匠", "敵役", "サポート", "その他"], index=["主人公", "ヒロイン", "ライバル", "親友", "師匠", "敵役", "サポート", "その他"].index(char_data_orig.get('role', 'その他')), key=f"edit_role_{char_to_edit}")
                    edited_char_details = st.text_area("詳細設定", value=char_data_orig.get('details', ''), key=f"edit_details_{char_to_edit}", height=300)

                    if st.button("変更を保存", key=f"save_char_{char_to_edit}"):
                        if edited_char_name not in project['characters'] or edited_char_name == char_to_edit:
                            project['characters'][edited_char_name] = {
                                'role': edited_char_role,
                                'details': edited_char_details
                            }
                            if edited_char_name != char_to_edit:
                                del project['characters'][char_to_edit]
                            
                            del st.session_state.editing_character
                            st.success(f"キャラクター「{edited_char_name}」を更新しました。")
                            st.rerun()
                        else:
                            st.warning(f"キャラクター「{edited_char_name}」は既に存在します。")

                    if st.button("キャンセル", key=f"cancel_char_{char_to_edit}"):
                        del st.session_state.editing_character
                        st.rerun()

        with tab3: # 世界観
            st.header("🗺️ 世界観設定")
            if current_writing_mode == 'manual' or current_writing_mode == 'hybrid':
                st.subheader("基本世界観設定")
                project['world_setting'] = st.text_area("世界観の詳細", value=project.get('world_setting', ''), height=300, help="物語の舞台となる世界の背景、ルール、特徴などを記述します。")

            if current_writing_mode == 'ai' or current_writing_mode == 'hybrid':
                st.subheader("🤖 AI 世界観生成")
                world_elements = st.text_area("世界観に追加したい要素（任意）", placeholder="例：魔法体系、国家間の関係、主要な産業...", height=80, key="world_elements_input")
                if st.button("🌍 AI世界観生成", key="generate_world_btn"):
                    if not is_api_key_set():
                        st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                    else:
                        with st.spinner("世界観生成中..."):
                            ai_world_setting = generate_ai_content("world_setting", project, {"world_elements": world_elements})
                            if not ai_world_setting.startswith("エラー"):
                                project['world_setting'] = ai_world_setting
                                st.success("世界観を生成しました！")
                                st.rerun()
                            else:
                                st.error(ai_world_setting)

            if project.get('world_setting'):
                with st.expander("🔧 世界観の推敲・修正 (AI)"):
                    world_modification = st.text_area("修正指示", placeholder="例：ファンタジー要素を強く、科学技術レベルを詳細に...", height=60, key="world_mod")
                    if st.button("🤖 世界観を修正", key="modify_world_btn") and world_modification:
                        if not is_api_key_set():
                            st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                        else:
                            with st.spinner("修正中..."):
                                modified_world = modify_content_with_ai(project['world_setting'], world_modification, "世界観")
                                if not modified_world.startswith("エラー"):
                                    st.session_state.modified_world = modified_world
                                    st.success("修正案が作成されました。内容を確認し、採用ボタンを押してください。")
                                    st.rerun()
                                else:
                                    st.error(modified_world)
                
                if 'modified_world' in st.session_state and st.session_state.modified_world:
                    st.write("#### 修正案の確認")
                    col_world_rev1, col_world_rev2 = st.columns(2)
                    with col_world_rev1:
                        st.write("**修正前**"); st.write(project['world_setting'])
                    with col_world_rev2:
                        st.write("**修正後**"); st.write(st.session_state.modified_world)
                    
                    if st.button("✅ 修正版を採用", key="accept_world_mod"):
                        project['world_setting'] = st.session_state.modified_world
                        del st.session_state.modified_world
                        st.success("修正版を採用しました！")
                        st.rerun()

        with tab4: # 執筆
            st.header("📖 執筆・原稿管理")
            
            execution_mode = st.radio("執筆方法を選択してください", ["📝 章ごと執筆", "📚 作品全体をAIで生成"], key="writing_tab_mode", horizontal=True)

            if execution_mode == "📝 章ごと執筆":
                st.subheader("章ごとの執筆")
                chapter_name = st.text_input("章のタイトル / 番号", value=project.get('current_chapter_name', ''))
                plot_outline = st.text_area("この章のプロット概要", value=project.get('current_chapter_plot', ''), height=100)
                target_length = st.text_input("目標文字数", value=project.get('current_chapter_length', '3000-5000字'))
                writing_style = st.selectbox("文体", ["三人称", "一人称"], index=0 if project.get('current_chapter_style', '三人称') == '三人称' else 1)

                if st.button("✍️ この章を執筆", key="write_chapter_btn"):
                    if not chapter_name or not plot_outline:
                        st.warning("章のタイトルとプロット概要を入力してください。")
                    else:
                        project['current_chapter_name'] = chapter_name
                        project['current_chapter_plot'] = plot_outline
                        project['current_chapter_length'] = target_length
                        project['current_chapter_style'] = writing_style

                        if not is_api_key_set():
                            st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                        else:
                            with st.spinner("執筆中..."):
                                chapter_content = generate_ai_content("chapter", project, {
                                    "chapter_name": chapter_name,
                                    "chapter_plot": plot_outline,
                                    "target_length": target_length,
                                    "writing_style": writing_style
                                })
                                if not chapter_content.startswith("エラー"):
                                    project['chapters'][chapter_name] = chapter_content
                                    st.success(f"「{chapter_name}」の執筆が完了しました！")
                                    st.rerun()
                                else:
                                    st.error(chapter_content)
                
                st.subheader("執筆済み章一覧")
                if not project['chapters']:
                    st.info("まだ章は執筆されていません。")
                else:
                    for chap_name, chap_content in project['chapters'].items():
                        with st.expander(f"📖 {chap_name}"):
                            st.text_area(f"{chap_name} の内容", value=chap_content, height=200, key=f"chapter_content_{chap_name}")
                            if st.button(f"{chap_name} をAIで修正・追記", key=f"edit_chapter_{chap_name}"):
                                st.session_state.editing_chapter_content = chap_content
                                st.session_state.editing_chapter_name = chap_name
                                st.rerun()

            elif execution_mode == "📚 作品全体をAIで生成":
                st.subheader("📚 作品全体をAIで生成")
                total_length = st.text_input("希望する総文字数", value=project.get('full_story_length', '10000-15000字'))
                chapter_count = st.text_input("希望する章数", value=project.get('full_story_chapters', '3-5章'))
                full_writing_style = st.selectbox("文体", ["三人称", "一人称"], index=0 if project.get('full_story_style', '三人称') == '三人称' else 1, key="full_story_style_select")
                
                if st.button("🎭 作品全体を生成", type="primary", key="generate_full_story_btn"):
                    if not is_api_key_set():
                        st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                    else:
                        with st.spinner("作品全体を生成中... 少々お待ちください。"):
                            full_story_content = generate_ai_content("full_story", project, {
                                "target_length": total_length,
                                "chapter_count": chapter_count,
                                "writing_style": full_writing_style
                            })
                            if not full_story_content.startswith("エラー"):
                                project['chapters'] = {"全体生成結果": full_story_content}
                                project['full_story_length'] = total_length
                                project['full_story_chapters'] = chapter_count
                                project['full_story_style'] = full_writing_style
                                
                                st.success("作品全体の生成が完了しました！「執筆済み章一覧」で確認できます。")
                                st.rerun()
                            else:
                                st.error(full_story_content)
            
            if 'editing_chapter_content' in st.session_state and st.session_state.editing_chapter_name:
                chapter_name_to_edit = st.session_state.editing_chapter_name
                original_content = st.session_state.editing_chapter_content
                
                with st.dialog(f"「{chapter_name_to_edit}」の内容を編集", key="edit_chapter_dialog"):
                    edited_chapter_content = st.text_area("編集内容", value=original_content, height=400, key=f"edit_chapter_text_{chapter_name_to_edit}")
                    
                    modification_instruction = st.text_area("AIによる修正指示（任意）", placeholder="例：この部分をもっと詳しく描写してほしい、セリフを変更してほしい...", height=80, key=f"edit_chapter_instruction_{chapter_name_to_edit}")
                    
                    col_edit_save, col_edit_cancel, col_edit_ai_modify = st.columns(3)
                    
                    with col_edit_save:
                        if st.button("変更を保存", key=f"save_chapter_edit_{chapter_name_to_edit}"):
                            project['chapters'][chapter_name_to_edit] = edited_chapter_content
                            del st.session_state.editing_chapter_content
                            del st.session_state.editing_chapter_name
                            st.success("章の内容を保存しました。")
                            st.rerun()
                    with col_edit_cancel:
                        if st.button("キャンセル", key=f"cancel_chapter_edit_{chapter_name_to_edit}"):
                            del st.session_state.editing_chapter_content
                            del st.session_state.editing_chapter_name
                            st.rerun()
                    with col_edit_ai_modify:
                        if st.button("🤖 AIで修正", key=f"ai_modify_chapter_{chapter_name_to_edit}") and modification_instruction:
                            if not is_api_key_set():
                                st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                            else:
                                with st.spinner("AIで修正中..."):
                                    modified_content = modify_content_with_ai(original_content, modification_instruction, "章の内容")
                                    if not modified_content.startswith("エラー"):
                                        st.session_state.editing_chapter_content = modified_content
                                        st.success("修正案が作成されました。内容を確認し、「変更を保存」または「キャンセル」を選択してください。")
                                        st.rerun()
                                    else:
                                        st.error(modified_content)

        with tab5: # 品質チェック
            st.header("🔍 品質チェック・診断")
            st.subheader("作品の総合診断")
            if st.button("📊 作品総合診断を実行", key="run_diagnosis_btn"):
                if not is_api_key_set():
                    st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                else:
                    with st.spinner("総合診断中..."):
                        diagnosis_prompt = f"""
あなたは経験豊富なライトノベルの編集者です。以下の作品設定とあらすじ、キャラクター情報を分析し、読者を惹きつけるレベルに達しているか、多角的な視点から評価・診断してください。

【作品基本情報】
ジャンル: {project.get('genre', '未設定')}
ターゲット読者: {project.get('target_audience', '未設定')}
テーマ: {project.get('theme', '未設定')}
あらすじ: {project.get('synopsis', '未設定')}
世界観: {project.get('world_setting', '未設定')[:1000]}
キャラクター（抜粋）:
{json.dumps({k:v.get('role') for k,v in list(project.get('characters', {}).items())[:5]}, indent=2, ensure_ascii=False)}

【評価項目】
1.  **作品の魅力・独自性**: どれだけ読者の興味を引き、他作品との差別化ができているか。
2.  **ストーリー展開**: プロットの面白さ、テンポ、伏線、クリフハンガーの適切さ。
3.  **キャラクターの魅力**: 主人公や主要キャラクターの造形の深さ、共感性、成長性。
4.  **世界観のリアリティ・魅力**: 設定の緻密さ、想像力、物語との整合性。
5.  **文章力・表現力**: 読みやすさ、描写の豊かさ、感情表現の巧みさ。
6.  **ターゲット読者への訴求力**: 設定や展開がターゲット層に響いているか。
7.  **全体的な完成度・商業性**: ライトノベルとして市場に受け入れられる可能性。

各項目について、5段階評価（★☆☆☆☆ ～ ★★★★★）で評価し、具体的な改善点を提案してください。最も改善が必要な点、そして作品の強みを明確にしてください。
"""
                        api_response = call_generative_api(diagnosis_prompt)
                        if not api_response['text'].startswith("エラー"):
                            st.session_state.diagnosis_result = api_response['text']
                            st.success("作品総合診断が完了しました。結果を表示します。")
                            st.rerun()
                        else:
                            st.error(api_response['text'])
            
            if 'diagnosis_result' in st.session_state:
                st.subheader("診断結果")
                st.markdown(st.session_state.diagnosis_result)

            st.markdown("---")
            st.subheader("ライトノベル要素チェックリスト")
            st.markdown("""
            ### 🎯 魅力的な作品のためのチェックリスト
            **基本要件**
            - ✅ 十分な文字数（例: 50,000文字以上）
            - ✅ 魅力的なキャラクター設定（主人公に共感できるか）
            - ✅ 読者を引き込む書き出し（冒頭数ページで興味を引くか）
            - ✅ 一貫性のある世界観と設定（矛盾がないか）
            - ✅ 読者層に響くテーマやメッセージ（ターゲットに刺さるか）
            
            **品質要件**
            - ✅ 完成度の高い文章力（読みやすいか、誤字脱字はないか）
            - ✅ 魅力的な描写力（情景、感情、心理描写など）
            - ✅ ストーリー展開のテンポ（飽きさせないか、盛り上がりがあるか）
            - ✅ キャラクターの魅力と成長（魅力的で、物語を通して変化するか）
            - ✅ 世界観の独自性・面白さ（魅力的で、物語に深みを与えているか）
            - ✅ テーマの掘り下げ（テーマが物語全体を通して描かれているか）
            - ✅ 読者の期待を超える要素（意外性、感動、興奮など）
            """)

        with tab6: # 分析・改善
            st.header("📊 分析・改善提案")
            st.subheader("🚀 作品をより良くするための改善提案")
            if st.button("💡 総合改善提案を生成", key="generate_improvement_btn"):
                if not is_api_key_set():
                    st.error(f"AI機能を利用するには、サイドバーで{st.session_state.selected_model_provider}のAPIキーを設定してください。")
                else:
                    with st.spinner("改善提案を生成中..."):
                        improvement_prompt = f"""
あなたはライトノベルの専門家であり、プロの編集者です。以下の作品情報を基に、読者にさらに愛される作品にするための具体的な改善提案を行ってください。

【作品情報】
ジャンル: {project.get('genre', '未設定')}
ターゲット読者: {project.get('target_audience', '未設定')}
テーマ: {project.get('theme', '未設定')}
あらすじ: {project.get('synopsis', '未設定')}
世界観: {project.get('world_setting', '未設定')[:1000]}
主要キャラクター（抜粋）:
{json.dumps({k:v.get('role') for k,v in list(project.get('characters', {}).items())[:5]}, indent=2, ensure_ascii=False)}

【改善提案の観点】
1.  **読者のエンゲージメント向上**: 読者が物語にさらに没入し、キャラクターに感情移入できるよう、どのような要素を加えるべきか。
2.  **ストーリーのフック強化**: プロットに更なる魅力を加えるためのアイデア（伏線、どんでん返し、葛藤の深化など）。
3.  **キャラクターアークの深化**: キャラクターに更なる深みや成長を与えるための要素。
4.  **世界観の活用**: 設定を物語の面白さにどう活かすか、深掘りすべき点。
5.  **テーマの強調**: 作品のテーマを読者に強く印象付けるための方法。
6.  **ライトノベルとしての独自性**: 他作品との差別化を図り、読者の記憶に残る作品にするための工夫。

これらの観点に基づき、具体的で実践的な改善策を提案してください。
"""
                        api_response = call_generative_api(improvement_prompt)
                        if not api_response['text'].startswith("エラー"):
                            st.session_state.improvement_suggestion = api_response['text']
                            st.success("改善提案の生成が完了しました。")
                            st.rerun()
                        else:
                            st.error(api_response['text'])

            if 'improvement_suggestion' in st.session_state:
                st.subheader("改善提案")
                st.markdown(st.session_state.improvement_suggestion)

    else:
        st.info("📁 左サイドバーから新規プロジェクトを作成するか、既存プロジェクトを選択してください。")
        st.markdown("""
        ## 🎯 使い方ガイド
        このツールは、あなたの物語作りを多角的にサポートします。
        
        ### 🚀 執筆スタイル
        - **🖊️ セルフ執筆**: あなたの言葉で物語を紡ぎます。AIはアイデア出しや推敲のお手伝いをします。
        - **🤖 AI執筆支援**: AIにたたき台を作成してもらい、それを基にあなたの創作を広げます。
        - **🔄 ハイブリッド**: 両方の良いところを組み合わせ、効率的にクオリティの高い作品を目指します。
        
        ### 📝 創作の流れ
        1.  **企画設定**: ジャンルやテーマ、キャラクター、世界観を考えます。
        2.  **執筆**: 章ごとに、または一気に物語を書き進めます。
        3.  **推敲・改善**: AIの助けを借りながら、より良い表現や展開を模索します。
        4.  **完成**: あなただけの素晴らしい物語を完成させましょう。
        """)

    # --- フッター ---
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <h4>🌟 あなたのアイデアを形にしよう</h4>
        <p>このツールが、あなたの素晴らしい創作活動の一助となれば幸いです。</p>
        <p><strong>継続は力なり。</strong> 楽しみながら創作を続けていきましょう。</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔑 API Key セットアップガイド", expanded=True):
        st.markdown("""
        ### 📝 AI機能を使うには (API Keyの取得方法)
        このツールのAI機能を利用するには、各AIサービスのAPIキーが必要です。
        
        1.  **利用したいAIモデルを選択**: サイドバーで「Gemini」「OpenAI」「Claude」から選びます。
        2.  **各公式サイトでAPIキーを取得**:
            -   **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey) からAPIキーを取得してください。Gemini 2.0 Flash など最新モデルもここで管理されます。
            -   **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys) からAPIキーを取得してください。
            -   **Claude**: [Anthropic Console](https://console.anthropic.com/dashboard) からAPIキーを取得してください。
        3.  **キーをアプリに設定**: 取得したキーを、サイドバーの対応する入力欄に貼り付けます。APIキーは機密情報ですので、公開しないようにご注意ください。
        
        **Streamlit Cloudをご利用の場合**:
        デプロイ後、Streamlit Cloudのアプリ設定画面にある「Secrets」セクションで、以下の形式でAPIキーを設定してください。
        ```toml
        GEMINI_API_KEY = "取得したGeminiのAPIキー"
        OPENAI_API_KEY = "取得したOpenAIのAPIキー"
        CLAUDE_API_KEY = "取得したClaudeのAPIキー"
        # APP_PASSWORD = "あなたの設定したパスワード" # もしアプリ全体にパスワードを設定する場合
        ```
        これにより、キーがコードに直接含まれることなく安全に管理され、アプリ内で利用できるようになります。
        
        **料金について**: 各AIサービスは無料利用枠を提供している場合がありますが、それを超えると利用量に応じた料金が発生します。必ず各公式サイトで最新の料金体系をご確認ください。
        """)

# --- アプリケーションの実行 ---
if not st.session_state.logged_in:
    # アプリ初回起動時、またはセッションが失われた場合
    if st.session_state.get('registered_username') is None:
        setup_user_view() # ユーザー登録画面を表示
    else:
        login_view() # 登録済みユーザーはログイン画面へ
else:
    main_app_view()