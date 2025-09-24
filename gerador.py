# -*- coding: utf-8 -*-

# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÃO INICIAL
# ==============================================================================
import os
import sys
import logging
import time
import unicodedata
from typing import Tuple, Dict, List, Optional

# Libs de terceiros (coloque no seu requirements.txt)
import pandas as pd
import matplotlib.pyplot as plt

# Tenta importar bibliotecas opcionais e define flags
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    PDF_ENABLED = True
except ImportError:
    PDF_ENABLED = False

try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    AGNO_ENABLED = True
except ImportError:
    AGNO_ENABLED = False

# ==============================================================================
# 2. CONSTANTES E CONFIGURAÇÕES GLOBAIS
# ==============================================================================
DEFAULT_COUNTRY = "Brazil"
COL_ENTITY = "Entity"
COL_YEAR = "Year"
COL_UNEMPLOYMENT = "Unemployment rate (%)"
COL_GDP_PER_CAPITA = "GDP per capita"
COL_INFLATION = "Inflation rate"
COL_CO2 = "CO2 emissions (kt)"
DEFAULT_YEARS = 5
CACHE_DIR = ".cache"
DEFAULT_MODEL_ID = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.9

# Mapeamento para encontrar arquivos de dados locais
EXACT_FILE_PATHS = {
    COL_UNEMPLOYMENT: "unemployment_rate_(%)/unemployment.csv",
    COL_GDP_PER_CAPITA: "gdp_per_capita/gdp.csv", 
    COL_INFLATION: "inflation_rate/inflation.csv",
    COL_CO2: "co2_emissions_(kt)/co2.csv",
}

# Palavras-chave para identificar indicadores nos arquivos e no texto da notícia
INDICATOR_KEYWORDS = {
    COL_UNEMPLOYMENT: ["unemployment", "desemprego", "jobless"],
    COL_GDP_PER_CAPITA: ["gdp", "pib", "per-capita", "capita", "gross domestic product"],
    COL_INFLATION: ["inflation", "inflacao", "cpi", "consumer price index"],
    COL_CO2: ["co2", "carbon", "emission", "carbono", "emissao", "emissões"],
}

# ==============================================================================
# 3. SETUP E FUNÇÕES UTILITÁRIAS
# ==============================================================================
def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

def _apply_env_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"): continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))
    except FileNotFoundError:
        pass

def load_env() -> None:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        _apply_env_file(env_path)
        logging.info(f".env carregado de: {env_path}")
    else:
        logging.debug("Arquivo .env não encontrado.")

def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Chave da OpenAI não encontrada! Defina a variável de ambiente OPENAI_API_KEY.")
    return key

def normalize_text(text: str) -> str:
    """Converte texto para minúsculas e remove acentos."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()

def _sanitize_filename(text: str) -> str:
    """Cria um nome de arquivo seguro a partir de um texto."""
    text_normalized = normalize_text(text)
    return ''.join(c for c in text_normalized.replace(" ", "_") if c.isalnum() or c == '_')

# ==============================================================================
# 4. CARREGAMENTO E PROCESSAMENTO DE DADOS LOCAIS
# ==============================================================================
def find_csv_files() -> Dict[str, str]:
    """Busca arquivos CSV na pasta .cache, por caminho exato e por palavras-chave."""
    found_files = {}
    if not os.path.exists(CACHE_DIR):
        logging.error(f"Pasta de dados '{CACHE_DIR}' não encontrada!")
        return found_files

    for indicator, rel_path in EXACT_FILE_PATHS.items():
        full_path = os.path.join(CACHE_DIR, rel_path)
        if os.path.exists(full_path):
            found_files[indicator] = full_path

    if len(found_files) < len(INDICATOR_KEYWORDS):
        for root, _, files in os.walk(CACHE_DIR):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    search_text = normalize_text(full_path)
                    for indicator, keywords in INDICATOR_KEYWORDS.items():
                        if indicator not in found_files and any(kw in search_text for kw in keywords):
                            found_files[indicator] = full_path
                            logging.info(f"Arquivo encontrado por keyword para '{indicator}': {os.path.basename(full_path)}")
                            break
                            
    logging.info(f"Mapeamento de arquivos de dados concluído: {len(found_files)} encontrados.")
    return found_files

def _normalize_dataframe(df: pd.DataFrame, target_col_name: str) -> pd.DataFrame:
    """Renomeia colunas de um DataFrame para um formato padrão ('Entity', 'Year', 'Valor')."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    rename_map = {}
    for col in df.columns:
        col_normalized = normalize_text(col)
        if 'entity' in col_normalized or 'country' in col_normalized or 'pais' in col_normalized:
            rename_map[col] = COL_ENTITY
        elif 'year' in col_normalized or 'ano' in col_normalized:
            rename_map[col] = COL_YEAR
    df = df.rename(columns=rename_map)

    keywords = INDICATOR_KEYWORDS.get(target_col_name, [])
    for col in df.columns:
        if col not in [COL_ENTITY, COL_YEAR] and any(kw in normalize_text(col) for kw in keywords):
            df = df.rename(columns={col: target_col_name})
            break
            
    required_cols = [COL_ENTITY, COL_YEAR, target_col_name]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    df_clean = df[required_cols].copy()
    for col_name in required_cols:
        if col_name != COL_ENTITY:
            df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')
            
    return df_clean.dropna()

def load_and_process_csv(file_path: str, target_col_name: str) -> Optional[pd.DataFrame]:
    """Tenta carregar um CSV de forma flexível, testando encodings e headers."""
    logging.info(f"Processando: {os.path.basename(file_path)} para '{target_col_name}'")
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        for header_row in range(5):
            try:
                df = pd.read_csv(file_path, header=header_row, encoding=encoding, on_bad_lines='skip', engine='python')
                df_normalized = _normalize_dataframe(df, target_col_name)
                if not df_normalized.empty:
                    logging.info(f"-> Sucesso na leitura e normalização.")
                    return df_normalized
            except Exception:
                continue
    logging.warning(f"-> Falha ao processar o arquivo: {os.path.basename(file_path)}")
    return None

def load_all_local_datasets() -> Dict[str, pd.DataFrame]:
    """Carrega todos os datasets a partir dos arquivos locais."""
    datasets = {}
    csv_files = find_csv_files()
    if not csv_files:
        return datasets
    
    for indicator, file_path in csv_files.items():
        dataset = load_and_process_csv(file_path, indicator)
        if dataset is not None:
            datasets[indicator] = dataset
            
    return datasets

# ==============================================================================
# 5. PREPARAÇÃO DE DADOS PARA ANÁLISE
# ==============================================================================
def prepare_country_last_n(df: pd.DataFrame, country: str, years: int, col_value: str) -> pd.DataFrame:
    """Filtra e prepara os dados de um país para os últimos N anos."""
    country_normalized = normalize_text(country)
    df['normalized_entity'] = df[COL_ENTITY].apply(normalize_text)
    subset = df[df['normalized_entity'].str.contains(country_normalized, na=False)].copy()
    
    if subset.empty:
        raise ValueError(f"Nenhum registro para '{country}' no dataset de '{col_value}'.")
    
    subset = subset.groupby(COL_YEAR, as_index=False)[col_value].mean()
    subset = subset.sort_values(COL_YEAR).tail(years)
    
    if subset.empty:
        raise ValueError(f"Sem dados válidos para '{country}' nos últimos {years} anos.")
    return subset.drop(columns=['normalized_entity'], errors='ignore')

def format_data_for_prompt(all_data: Dict[str, pd.DataFrame], country: str, years: int) -> str:
    """Formata um resumo de todos os dados disponíveis para enviar à IA."""
    summary_parts = []
    for key, df in all_data.items():
        try:
            subset = prepare_country_last_n(df, country, years, col_value=key)
            if not subset.empty:
                years_list = subset[COL_YEAR].astype(int).tolist()
                rates = subset[key].astype(float).round(2).tolist()
                summary = f"Dados de {key.lower()} para {country}:\n" + "\n".join([f"- {y}: {r}" for y, r in zip(years_list, rates)])
                summary_parts.append(summary)
        except ValueError as e:
            logging.debug(f"Não foi possível formatar dados de '{key}': {e}")
    return "\n\n".join(summary_parts)

# ==============================================================================
# 6. IA E GERAÇÃO DE CONTEÚDO
# ==============================================================================
class SimulatedAgent:
    """Agente simulado caso a biblioteca 'agno' não esteja disponível."""
    def run(self, prompt: str) -> str:
        return "# Análise Simulada\n\nEste é um texto gerado pelo agente simulado. Instale a biblioteca 'agno' e configure sua chave de API para análises reais."

def build_agent(api_key: str, model_id: str, temperature: float) -> 'Agent':
    if not AGNO_ENABLED: return SimulatedAgent()
    try:
        if "warning" in logging.Logger.manager.loggerDict:
            logging.getLogger("warning").setLevel(logging.ERROR)
        return Agent(
            model=OpenAIChat(id=model_id, api_key=api_key, temperature=temperature),
            description="Você é um jornalista de dados econômicos. Sua tarefa é analisar dados e escrever uma notícia clara e objetiva, identificando as tendências mais importantes.",
            markdown=True,
        )
    except Exception as e:
        logging.error(f"Erro ao criar agente 'agno': {e}. Usando agente simulado.")
        return SimulatedAgent()

def build_prompt_news(country: str, data_summary: str, pauta: Optional[str]) -> str:
    """Constrói o prompt para a geração da notícia."""
    pauta_text = f"A pauta principal é: '{pauta}'." if pauta else "A pauta é automática, encontre o ângulo mais interessante nos dados."
    return (
        f"Escreva uma notícia em português do Brasil sobre o cenário econômico de {country}, baseada nos seguintes dados:\n\n"
        f"{data_summary}\n\n"
        f"{pauta_text}\n\n"
        "Estrutura obrigatória: Título; Subtítulo; Lide (1º parágrafo); Corpo (2-4 parágrafos); Conclusão. "
        "Seja factual e use os números para suportar sua análise. Não invente dados."
    )

def get_mentioned_indicators(news_text: str, available_indicators: List[str]) -> List[str]:
    """Analisa o texto da notícia para identificar quais indicadores foram mencionados."""
    logging.info("Analisando notícia para identificar indicadores para o gráfico...")
    text_normalized = normalize_text(news_text)
    mentioned = [ind for ind in available_indicators if any(kw in text_normalized for kw in INDICATOR_KEYWORDS.get(ind, []))]
    
    if not mentioned:
        logging.warning("Nenhum indicador detectado na notícia. O gráfico mostrará todos os dados disponíveis como fallback.")
        return available_indicators

    logging.info(f"Indicadores mencionados detectados: {mentioned}")
    return mentioned

# ***** NOVA FUNÇÃO ADICIONADA *****
def get_indicators_from_pauta(pauta: str, available_indicators: List[str]) -> List[str]:
    """Analisa a pauta para extrair os indicadores principais para o gráfico."""
    if not pauta:
        return []
    
    logging.info(f"Analisando pauta '{pauta}' para extrair indicadores do gráfico...")
    pauta_normalized = normalize_text(pauta)
    mentioned = [ind for ind in available_indicators if any(kw in pauta_normalized for kw in INDICATOR_KEYWORDS.get(ind, []))]
    
    if mentioned:
        logging.info(f"Indicadores encontrados diretamente na pauta: {mentioned}")
    
    return mentioned

def _to_text(result) -> str:
    if result is None: return ""
    if isinstance(result, str): return result
    if hasattr(result, "content"): return str(getattr(result, "content", ""))
    return str(result)

def generate_content(agent: 'Agent', prompt_text: str) -> str:
    try:
        result = agent.run(prompt_text)
        return _to_text(result)
    except Exception as e:
        raise RuntimeError(f"Falha ao gerar conteúdo com o agente: {e}") from e

# ==============================================================================
# 7. SAÍDA E VISUALIZAÇÃO
# ==============================================================================
def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    logging.info(f"Arquivo de texto salvo: {path}")

def save_text_as_pdf(text: str, path: str) -> None:
    if not PDF_ENABLED: return
    try:
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(line, styles['Normal']) for line in text.split('\n') if line.strip()]
        doc.build(story)
        logging.info(f"PDF salvo: {path}")
    except Exception as e:
        logging.error(f"Falha ao gerar PDF: {e}")

def plot_and_save(country_data: pd.DataFrame, country: str, output_path: str) -> None:
    """Cria e salva uma grade de gráficos, adaptando o layout ao número de indicadores."""
    indicators = [col for col in country_data.columns if col not in [COL_ENTITY, COL_YEAR]]
    if not indicators:
        logging.warning("Nenhum dado para plotar. Gráfico não será gerado.")
        return

    num_indicators = len(indicators)
    nrows = 1 if num_indicators <= 2 else 2
    ncols = 1 if num_indicators == 1 else 2
    figsize_h = 5 if nrows == 1 else 10
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, figsize_h), squeeze=False)
    axes = axes.flatten()

    for i, indicator in enumerate(indicators):
        ax = axes[i]
        ax.plot(country_data[COL_YEAR].astype(int), country_data[indicator].astype(float), marker='o', linestyle='-')
        ax.set_title(indicator, fontsize=12, fontweight='bold')
        ax.set_ylabel("Valor")
        ax.set_xlabel("Ano")
        ax.grid(True, alpha=0.5)

    for j in range(num_indicators, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Indicadores Mencionados na Notícia - {country}", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    logging.info(f"Gráfico salvo: {output_path}")
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)

# ==============================================================================
# 8. FUNÇÃO DE EXECUÇÃO PRINCIPAL
# ==============================================================================
def main() -> None:
    setup_logging()
    try:
        # --- 1. SETUP INICIAL ---
        load_env()
        api_key = get_openai_key()
        logging.info(f"Chave da API: {api_key[:4]}...")
        agent = build_agent(api_key, DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE)

        # --- 2. INPUT DO USUÁRIO ---
        country = input(f"País para análise? (Enter para {DEFAULT_COUNTRY}): ").strip() or DEFAULT_COUNTRY
        pauta = input("Pauta para a análise? (Enter para automática): ").strip()

        # --- 3. CARREGAMENTO E VALIDAÇÃO DOS DADOS ---
        all_data = load_all_local_datasets()
        if not all_data:
            raise ValueError("Nenhum dataset foi carregado. Verifique os arquivos na pasta .cache.")

        # --- 4. GERAÇÃO DA NOTÍCIA ---
        data_summary = format_data_for_prompt(all_data, country, DEFAULT_YEARS)
        if not data_summary:
            raise ValueError(f"Não foi possível gerar resumo de dados para '{country}'. Verifique os arquivos CSV e o nome do país.")

        prompt_news = build_prompt_news(country, data_summary, pauta)
        news_text = generate_content(agent, prompt_news)
        print("\n---\n🗞️ Notícia Gerada:\n---\n" + news_text)
        
        # --- 5. PROCESSAMENTO E FILTRAGEM PARA O GRÁFICO (LÓGICA APRIMORADA) ---
        indicators_to_plot = get_indicators_from_pauta(pauta, list(all_data.keys()))

        if not indicators_to_plot:
            logging.info("Nenhum indicador específico na pauta, analisando o texto completo da notícia como fallback.")
            indicators_to_plot = get_mentioned_indicators(news_text, list(all_data.keys()))
        
        data_for_plot = pd.DataFrame()
        for indicator in indicators_to_plot:
            try:
                if indicator in all_data:
                    subset = prepare_country_last_n(all_data[indicator], country, DEFAULT_YEARS, indicator)
                    if data_for_plot.empty:
                        data_for_plot = subset
                    else:
                        data_for_plot = pd.merge(data_for_plot, subset, on=COL_YEAR, how='outer')
            except ValueError as e:
                logging.warning(f"Não foi possível preparar dados para '{indicator}': {e}")
        
        # --- 6. SALVAMENTO DOS ARTEFATOS (TEXTOS, PDF, GRÁFICO) ---
        file_prefix = f"{_sanitize_filename(country)}_{_sanitize_filename(pauta) or 'analise'}"
        
        save_text(news_text, f"{file_prefix}_noticia.txt")
        if PDF_ENABLED: save_text_as_pdf(news_text, f"{file_prefix}_noticia.pdf")
        
        if not data_for_plot.empty and len(data_for_plot.columns) > 1:
            plot_and_save(data_for_plot, country, f"grafico_{file_prefix}.png")
        else:
            logging.warning("DataFrame para plotagem está vazio ou inválido. Gráfico não será gerado.")

        # --- 7. GERAÇÃO DO POST DE LINKEDIN ---
        logging.info("Gerando post para LinkedIn...")
        prompt_post = (f"Com base na notícia, resuma os principais pontos em um post para LinkedIn. Use bullets e hashtags.\n\nNotícia: {news_text}")
        post_text = generate_content(agent, prompt_post)
        print("\n---\n📰 Post para LinkedIn:\n---\n" + post_text)
        
        save_text(post_text, f"{file_prefix}_post.txt")
        if PDF_ENABLED: save_text_as_pdf(post_text, f"{file_prefix}_post.pdf")
        
        print("\n✅ Processo concluído com sucesso!")

    except Exception as e:
        logging.error(f"Ocorreu um erro fatal: {e}")
        sys.exit(1)

# ==============================================================================
# 9. PONTO DE ENTRADA DO SCRIPT
# ==============================================================================
if __name__ == "__main__":
    main()