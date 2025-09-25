# 🤖 Agente IA NEWS

Um agente de IA desenvolvido em Python para analisar dados socioeconômicos do Brasil, gerar insights com a API da OpenAI e criar visualizações de dados.

---

## 📄 Sobre o Projeto

O Agente IA NEWS foi criado para automatizar o processo de análise de dados e geração de conteúdo. O script principal, `gerador.py`, coleta e processa indicadores econômicos e sociais do Brasil a partir de arquivos `.csv` e utiliza a biblioteca `matplotlib` para criar gráficos.

O grande diferencial do projeto é o uso da **API da OpenAI** para interpretar os dados e os gráficos, gerando análises textuais coesas e informativas, como se fossem uma notícia ou um post de análise.

## ✨ Funcionalidades Principais

* **Análise de Múltiplos Indicadores:** Processa dados de CO₂, PIB, inflação e desemprego.
* **Geração de Texto com IA:** Usa um modelo de linguagem avançado para criar análises escritas.
* **Criação de Gráficos:** Gera visualizações dos dados processados usando `matplotlib`.
* **Gestão de Dependências:** Utiliza ambientes virtuais (`.venv`) e um arquivo `requirements.txt` para fácil instalação.
* **Segurança:** Configurado com `.gitignore` para proteger dados sensíveis como chaves de API (`.env`).

## 🚀 Como Começar

Siga os passos abaixo para configurar e executar o projeto em sua máquina local.

### Pré-requisitos

* [Python 3.9+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)

### Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/luisleoncio86/agente-IA-NEWS.git](https://github.com/luisleoncio86/agente-IA-NEWS.git)
    ```

2.  **Navegue até a pasta do projeto:**
    ```bash
    cd agente-IA-NEWS
    ```

3.  **Crie e ative um ambiente virtual:**
    ```bash
    # Crie o ambiente
    python -m venv .venv

    # Ative o ambiente (Windows PowerShell)
    .\.venv\Scripts\Activate.ps1
    ```

4.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure sua chave de API:**
    * Crie um arquivo chamado `.env` na pasta principal do projeto.
    * Dentro deste arquivo, adicione sua chave da OpenAI no seguinte formato:
        ```
        OPENAI_API_KEY="sua_chave_secreta_da_openai_aqui"
        ```

### Como Usar

Com o ambiente ativado e configurado, execute o script principal para gerar a análise, o texto e o gráfico:
```bash
python gerador.py

O resultado será salvo nos arquivos de saída especificados no código.

🛠️ Tecnologias Utilizadas
Python: Linguagem principal do projeto.

Pandas: Para manipulação e análise de dados.

Matplotlib: Para a criação dos gráficos.

OpenAI API: Para a geração dos insights em texto.

👨‍💻 Autor
Luis Leoncio - GitHub

---

### Como Adicionar ao seu GitHub:

1.  No VS Code, crie um novo arquivo na pasta principal chamado `README.md`.
2.  Copie e cole todo o conteúdo acima nesse arquivo.
3.  Salve o arquivo.
4.  No terminal, rode os seguintes comandos para enviá-lo:
    ```bash
    git add README.md
    git commit -m "Adiciona README.md detalhado ao projeto"
    git push
    ```
