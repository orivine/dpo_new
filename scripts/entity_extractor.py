import os
import json
import logging
import time
import hashlib
from typing import List, Dict, Union, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)    # logger.info才能显示

class EntityExtractor:
    """使用大语言模型抽取文本中的关键实体"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_retries: int = 3,
        retry_delay: int = 2,
        retry_min: int = 1,
        retry_max: int = 10,
        offline_mode: bool = False,
        cache_dir: str = "entity_cache",
    ):
        """
        初始化实体抽取器
        
        Args:
            api_key: API密钥，如果为None，则从环境变量MOONSHOT_API_KEY中获取
            base_url: API基础URL
            model: 使用的模型名称
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            retry_min: 最小重试延迟
            retry_max: 最大重试延迟
            offline_mode: 是否使用离线模式（不调用API）
            cache_dir: 实体缓存目录
        """
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not self.api_key and not offline_mode:
            print("警告: API密钥未提供，切换到离线模式")
            offline_mode = True
        
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_min = retry_min
        self.retry_max = retry_max
        self.offline_mode = offline_mode
        
        # 设置缓存相关属性
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "entity_cache.json")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """加载实体缓存"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载实体缓存失败: {str(e)}")
                return {}
        return {}
    
    def _save_cache(self):
        """保存实体缓存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存实体缓存失败: {str(e)}")
    
    def _generate_cache_key(self, text: str) -> str:
        """生成缓存键，使用文本的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def extract_entities(self, text: str) -> List[str]:
        """
        从文本中抽取关键实体，优先从缓存获取
        
        Args:
            text: 输入文本
            
        Returns:
            关键实体列表
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(text)
        
        # 检查缓存中是否存在
        if cache_key in self.cache:
            # logger.info(f"缓存中已存在实体列表: {cache_key[:10]}...")     # 日志：实体提取
            return self.cache[cache_key]
        
        # 如果处于离线模式，使用简单的规则提取关键词
        if self.offline_mode:
            entities = self._extract_entities_offline(text)
            # 存入缓存
            self.cache[cache_key] = entities
            self._save_cache()
            return entities
        
        # prompt = f"""请从以下文本中提取出关键的命名实体，包括但不限于人名、地名、组织机构名、时间、专有名词等。
        # 只返回实体列表，每个实体一行，不要有其他说明或格式，例如"1. "或"- "等前缀。"""
            
        # prompt = f"""请认真阅读以下文本，并根据文本的中心含义，从以下文本中提取出最有代表性的关键命名实体，包括但不限于人名、地名、组织机构名、时间、专有名词等。
        # 只返回实体列表，每个实体一行，提取出的实体需要和原文语言一致。不要有其他说明或格式，例如"1. "或"- "等前缀。
        
        # 文本：{text}
        
        # 关键实体："""

        print(f"提取实体的文本: {text}")

        prompt = f"""Please carefully read the text enclosed within <text></text> and extract the most representative named entities based on the central meaning of the text. These entities may include but are not limited to person, place, organization, events, proper nouns and descriptive phrases.
        Only return the list of entities, with each entity on a separate line. Please try not to return any entire sentence. The extracted entities should be in the same language as the original text. Do not include any additional explanations or formatting, such as prefixes like "1. " or "- ".

        Text: 
        <text>  
        {text}  
        </text>  
        Key entities:
        """

        # prompt = f"""  Analyze the text enclosed in <text></text> and extract semantically representative words and phrases in the text.
        # **Prioritize entities:**   
        # - People (e.g., scientists, public figures)  
        # - Organizations (e.g., companies, institutions, governments)  
        # - Locations (e.g., cities, countries, landmarks)  
        # - Events (e.g., Dates, wars, elections)  
        # - Key terms (e.g., things, laws, products, technologies)  
        # - Descriptive phrases (e.g., beautiful place, the most famous speaker, etc.)

        # **Rules:**  
        # 1. Return only the entities one per line, in their original language.  
        # 2. No prefixes, numbering, or additional explanations (e.g., avoid "1. ", "- ", or notes).  
        # 3. If no proper named entities found in the text, return the word "NoEntity".

        # <text>  
        # {text}  
        # </text>  

        # Key entities:  
        # """  

#         prompt = f"""  
# Analyze the text enclosed in <text></text> and extract **semantically representative named entities** that are critical to understanding the text's core meaning. 
# **Prioritize entities:**   
# - People (e.g., scientists, public figures)  
# - Organizations (e.g., companies, institutions, governments)  
# - Locations (e.g., cities, countries, landmarks)  
# - Special events (e.g., Dates, wars, elections)  
# - Key terms (e.g., titles, laws, products, technologies)  
# - Degree-modifying phrases (e.g., higher place, the most famous speaker, etc.)

# **Rules:**  
# 1. Return **only the entities**, one per line, in their original language.  
# 2. Exclude weak or unclear matches especially stopwords.  
# 3. **No prefixes, numbering, or additional explanations** (e.g., avoid "1. ", "- ", or notes).  
# 4. If no semantically representative named entities found in the text, return the word "NoEntity".

# <text>  
# {text}  
# </text>  

# Key entities:  
# """  
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            entities = [entity.strip() for entity in content.split("\n") if entity.strip()]
            
            # 过滤掉空实体和可能的编号前缀
            clean_entities = []
            for entity in entities:
                # 移除可能的序号前缀，如"1. "、"- "等
                if entity.startswith(("- ", "• ", "#")):
                    entity = entity[2:].strip()
                elif len(entity) > 2 and entity[0].isdigit() and entity[1:3] in (". ", ": "):
                    entity = entity[3:].strip()
                
                if entity:
                    clean_entities.append(entity)
            
            # 存入缓存
            self.cache[cache_key] = clean_entities
            self._save_cache()
            
            return clean_entities
            
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            # 如果API调用失败，切换到离线模式
            print(f"API调用失败: {str(e)}，切换到离线模式")
            entities = self._extract_entities_offline(text)
            
            # 存入缓存
            self.cache[cache_key] = entities
            self._save_cache()
            
            return entities
    
    def _extract_entities_offline(self, text: str) -> List[str]:
        """
        离线模式下使用简单规则提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            关键实体列表
        """
        # 常见的专有名词和关键词（示例）
        common_entities = [
            "AI", "机器学习", "深度学习", "神经网络", "大模型"
        ]
        
        # 检查文本中是否包含这些实体
        found_entities = []
        for entity in common_entities:
            if entity.lower() in text.lower():
                found_entities.append(entity)
        
        # 如果找不到任何实体，提取一些长词作为实体（简单启发式方法）
        if not found_entities:
            words = text.split()
            found_entities = [word for word in words if len(word) > 6 and word[0].isupper()][:5]
        
        return found_entities[:10]  # 限制数量
    
    def create_entity_masks(
        self, 
        tokenizer, 
        prompt: str, 
        chosen: str, 
        rejected: str
    ) -> Dict:
        """
        创建实体掩码
        
        Args:
            tokenizer: 分词器
            prompt: 提示文本
            chosen: 正例回答
            rejected: 负例回答
            
        Returns:
            掩码字典，包含chosen_mask、rejected_mask和entities
        """
        # 提取关键实体
        entities = self.extract_entities(prompt)
        if not entities:
            logger.warning(f"未能从文本中提取到关键实体: {prompt[:100]}...")
            # 创建与文本长度匹配的空掩码，而不是返回None
            chosen_tokens = tokenizer.tokenize(chosen)
            rejected_tokens = tokenizer.tokenize(rejected)
            chosen_mask = [0] * len(chosen_tokens)
            rejected_mask = [0] * len(rejected_tokens)
            
            return {
                "chosen_mask": chosen_mask,
                "rejected_mask": rejected_mask,
                "entities": [],
                "chosen_highlight": chosen,
                "rejected_highlight": rejected
            }
        
        # logger.info(f"提取到的关键实体: {entities}")     # 日志：实体提取
        
        # 为chosen和rejected创建掩码
        chosen_mask = self._create_mask_for_text(tokenizer, chosen, entities)
        rejected_mask = self._create_mask_for_text(tokenizer, rejected, entities)
        
        # 创建高亮的文本表示
        chosen_highlight = self.create_highlighted_text(tokenizer, chosen, chosen_mask)
        rejected_highlight = self.create_highlighted_text(tokenizer, rejected, rejected_mask)
        
        return {
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
            "entities": entities,  # 返回提取的实体列表
            "chosen_highlight": chosen_highlight,
            "rejected_highlight": rejected_highlight
        }
    
    def _create_mask_for_text(self, tokenizer, text: str, entities: List[str]) -> List[int]:
        """
        为文本创建实体掩码
        
        Args:
            tokenizer: 分词器
            text: 文本
            entities: 实体列表
            
        Returns:
            掩码列表，1表示token是关键实体的一部分，0表示不是
        """
        # 获取文本的token ids
        tokens = tokenizer.tokenize(text)
        
        # 初始化掩码，所有位置为0
        mask = [0] * len(tokens)
        
        # 对于每个实体，找到它在token中的位置并标记
        text_lower = text.lower()
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in text_lower:
                # 找出实体在原文中的所有位置
                start_pos = 0
                while True:
                    start_idx = text_lower.find(entity_lower, start_pos)
                    if start_idx == -1:
                        break
                    
                    end_idx = start_idx + len(entity)
                    
                    # 将实体转换为token，以便找到对应的token位置
                    char_to_token = tokenizer.encode(text[:end_idx], add_special_tokens=False)
                    if len(char_to_token) > 0:
                        # 找到实体对应的token位置范围
                        entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
                        entity_token_len = len(entity_tokens)
                        
                        # 获取实体结束位置对应的token索引
                        end_token_idx = len(char_to_token)
                        
                        # 获取实体开始位置对应的token索引
                        start_token_idx = max(0, end_token_idx - entity_token_len)
                        
                        # 标记掩码
                        for i in range(start_token_idx, min(end_token_idx, len(mask))):
                            mask[i] = 1
                    
                    start_pos = end_idx
        
        return mask
        
    def create_highlighted_text(self, tokenizer, text: str, mask: List[int]) -> str:
        """
        创建高亮显示的文本表示，方便查看掩码应用的位置
        
        Args:
            tokenizer: 分词器
            text: 原始文本
            mask: 掩码列表
            
        Returns:
            带有高亮标记的文本
        """
        if not mask:
            return text
            
        tokens = tokenizer.tokenize(text)
        highlighted_tokens = []
        
        for i, (token, is_entity) in enumerate(zip(tokens, mask)):
            if is_entity:
                highlighted_tokens.append(f"<___[{token}]___>")  # 突出表示
            else:
                highlighted_tokens.append(token)
        
        # 尝试将token重新合成文本（这只是个近似，可能与原文有差异）
        highlighted_text = tokenizer.convert_tokens_to_string(highlighted_tokens)
        return highlighted_text 