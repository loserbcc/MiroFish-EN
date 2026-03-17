"""
Configuration management
Loads configuration uniformly from the .env file in the project root directory
"""

import os
from dotenv import load_dotenv

# Load .env file from project root directory
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env)
else:
    # If no .env in root directory, try loading environment variables (for production)
    load_dotenv()

# Graphiti requires OPENAI_* environment variables, map from LLM_*
# Only map when not explicitly set, to avoid overriding user's explicit configuration
if not os.environ.get('OPENAI_API_KEY') and os.environ.get('LLM_API_KEY'):
    os.environ['OPENAI_API_KEY'] = os.environ['LLM_API_KEY']
if not os.environ.get('OPENAI_BASE_URL') and os.environ.get('LLM_BASE_URL'):
    os.environ['OPENAI_BASE_URL'] = os.environ['LLM_BASE_URL']


class Config:
    """Flask configuration class"""

    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON configuration - Disable ASCII escaping to display non-ASCII characters directly (instead of \uXXXX format)
    JSON_AS_ASCII = False

    # LLM configuration (unified OpenAI format)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')

    # Zep configuration
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    ZEP_BACKEND = os.environ.get('ZEP_BACKEND', 'cloud')  # 'cloud' | 'graphiti'

    # Graphiti / Neo4j configuration (for local deployment)
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing configuration
    DEFAULT_CHUNK_SIZE = 500  # Default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # Default overlap size

    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform available actions configuration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured")
        # Validate configuration based on backend type
        if cls.ZEP_BACKEND == 'cloud':
            if not cls.ZEP_API_KEY:
                errors.append("ZEP_API_KEY is not configured (required when ZEP_BACKEND=cloud)")
        elif cls.ZEP_BACKEND == 'graphiti':
            if not all([cls.NEO4J_URI, cls.NEO4J_USER, cls.NEO4J_PASSWORD]):
                errors.append("Neo4j configuration is incomplete (required when ZEP_BACKEND=graphiti)")
        return errors
