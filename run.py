import os
import fitz  # PyMuPDF
import numpy as np
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Carrega variáveis de ambiente (sua chave de API no .env)
load_dotenv()

# Inicializa os clientes de IA
ia_client = Groq()
gerador_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# =====================================================
# FUNÇÕES UTILITÁRIAS
# =====================================================

def calcular_similaridade(vetor_a, vetor_b):
    """Calcula a similaridade do cosseno entre dois vetores."""
    produto_escalar = np.dot(vetor_a, vetor_b)
    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)
    return produto_escalar / (norma_a * norma_b)

def ler_arquivo_txt(caminho_arquivo):
    """Lê o conteúdo de um arquivo de texto."""
    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        return arquivo.read()

def ler_arquivo_pdf(caminho_arquivo):
    """Extrai todo o texto de um arquivo PDF."""
    texto_completo = ""
    documento = fitz.open(caminho_arquivo)
    for pagina in documento:
        texto_completo += pagina.get_text()
    return texto_completo

def criar_chunks(texto, tamanho_maximo=500):
    """Divide um texto longo em pedaços menores (chunks)."""
    lista_chunks = []
    for i in range(0, len(texto), tamanho_maximo):
        pedaco = texto[i : i + tamanho_maximo]
        lista_chunks.append(pedaco)
    return lista_chunks

# =====================================================
# RESOLUÇÃO DOS EXERCÍCIOS
# =====================================================

def exercicio_1_e_2_ler_txt():
    print("\n--- EXTRAÇÃO E CHUNKING DE TXT ---")
    texto_txt = ler_arquivo_txt("docs/regulamento_empresa_base_dados.txt")
    print("Texto extraído com sucesso. Primeiros 100 caracteres:")
    print(texto_txt[:100] + "...\n")
    
    chunks = criar_chunks(texto_txt, 500)
    print(f"Quantidade de chunks gerados: {len(chunks)}")
    print("\nPrimeiro chunk:\n", chunks[0])
    print("\nÚltimo chunk:\n", chunks[-1])

def exercicio_1_e_2_ler_pdf():
    print("\n--- EXTRAÇÃO E CHUNKING DE PDF ---")
    texto_pdf = ler_arquivo_pdf("docs/teste.pdf")
    print("Texto extraído com sucesso. Primeiros 100 caracteres:")
    print(texto_pdf[:100] + "...\n")
    
    chunks = criar_chunks(texto_pdf, 500)
    print(f"Quantidade de chunks gerados: {len(chunks)}")
    print("\nPrimeiro chunk:\n", chunks[0])

def exercicio_3_gerar_embeddings():
    print("\n--- GERANDO EMBEDDINGS LOCAIS ---")
    frases_teste = [
        "A inteligência artificial está revolucionando a tecnologia.",
        "O aprendizado de máquina é um subcampo importante.",
        "Bancos de dados relacionais usam tabelas."
    ]
    
    vetores = gerador_embeddings.encode(frases_teste)
    print("Vetor da primeira frase gerado com sucesso!")
    print(f"Tamanho do vetor (dimensões): {len(vetores[0])}")
    print("Amostra dos 5 primeiros números do vetor:")
    print(vetores[0][:5])

def exercicio_4_busca_semantica():
    print("\n--- BUSCA SEMÂNTICA EM MEMÓRIA ---")
    conhecimento = [
        "O pão de queijo é originário de Minas Gerais.",
        "A Terra gira em torno do Sol.",
        "A capital da França é Paris.",
        "O aprendizado supervisionado utiliza dados com respostas corretas.",
        "Bancos de dados relacionais usam a linguagem SQL."
    ]
    
    vetores_conhecimento = gerador_embeddings.encode(conhecimento)
    
    pergunta_usuario = input("\nDigite sua pergunta: ")
    vetor_pergunta = gerador_embeddings.encode(pergunta_usuario)
    
    resultados = []
    for i, vetor_base in enumerate(vetores_conhecimento):
        pontuacao = calcular_similaridade(vetor_pergunta, vetor_base)
        resultados.append((pontuacao, conhecimento[i]))
        
    resultados.sort(reverse=True)
    
    print("\nFato mais relevante encontrado no banco:")
    print(f"-> {resultados[0][1]} (Score: {resultados[0][0]:.4f})")

def exercicio_5_rag_simples():
    print("\n--- RAG SIMPLES COM FATOS ---")
    conhecimento = [
        "Imortalidade biológica: Lagostas produzem uma enzima que repara suas células.",
        "Tatus à prova de balas: A couraça dos tatus é altamente resistente.",
        "Tempo dos roedores: Animais pequenos percebem o tempo em câmera lenta."
    ]
    
    vetores_conhecimento = gerador_embeddings.encode(conhecimento)
    pergunta_usuario = input("\nDigite sua pergunta: ")
    vetor_pergunta = gerador_embeddings.encode(pergunta_usuario)
    
    resultados = []
    for i, vetor_base in enumerate(vetores_conhecimento):
        pontuacao = calcular_similaridade(vetor_pergunta, vetor_base)
        resultados.append((pontuacao, conhecimento[i]))
        
    resultados.sort(reverse=True)
    contexto_recuperado = resultados[0][1]
    
    prompt_sistema = f"""
    Você é um assistente estrito. Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE no contexto abaixo.
    Se a resposta não estiver no contexto, diga: 'Não possuo essa informação.'
    
    CONTEXTO: {contexto_recuperado}
    PERGUNTA: {pergunta_usuario}
    """
    
    print("\nEnviando para a IA...")
    resposta_ia = ia_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt_sistema}]
    )
    
    print("\nRESPOSTA GERADA:")
    print(resposta_ia.choices[0].message.content)

def exercicio_6_pipeline_completo():
    print("\n--- PIPELINE RAG COMPLETO COM CHROMADB ---")
    
    # 1. Ingestão e Chunking
    print("1. Lendo arquivo e criando chunks...")
    texto_base = ler_arquivo_txt("docs/regulamento_empresa_base_dados.txt")
    chunks_documento = criar_chunks(texto_base, 500)
    
    # 2. Indexação com ChromaDB
    print("2. Gerando embeddings e salvando no Banco Vetorial...")
    # O ChromaDB exige que os embeddings sejam listas comuns do Python, por isso o .tolist()
    vetores_documento = gerador_embeddings.encode(chunks_documento).tolist()
    
    # Configurando o banco de dados na pasta local 'chroma_data'
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    colecao = chroma_client.get_or_create_collection(
        name="regulamento_empresa",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Criando IDs únicos para cada chunk (doc_0, doc_1, etc.)
    ids = [f"chunk_{i}" for i in range(len(chunks_documento))]
    
    # Salvando no banco
    colecao.upsert(
        ids=ids,
        documents=chunks_documento,
        embeddings=vetores_documento
    )
    
    # 3. Busca (Retrieval)
    pergunta_usuario = input("\nFaça uma pergunta sobre o regulamento da empresa: ")
    vetor_pergunta = gerador_embeddings.encode([pergunta_usuario]).tolist()
    
    # Buscando os 2 trechos mais parecidos direto no banco
    resultados = colecao.query(
        query_embeddings=vetor_pergunta,
        n_results=5 
    )
    
    melhores_chunks = resultados["documents"][0]
    contexto_final = "\n\n---\n\n".join(melhores_chunks)
    
    # 4. Geração (Generation)
    prompt_final = f"""
    Atue como o RH e Suporte de TI da Alpha Soluções Tecnológicas.
    Baseado EXCLUSIVAMENTE nas diretrizes abaixo, responda à dúvida do funcionário de forma clara e profissional.
    
    DIRETRIZES DA EMPRESA (Contexto):
    {contexto_final}
    
    DÚVIDA DO FUNCIONÁRIO:
    {pergunta_usuario}
    """
    
    print("\nGerando resposta baseada no regulamento (via LLM)...")
    resposta_ia = ia_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt_final}]
    )
    
    print("\nRESPOSTA DA EMPRESA:")
    print(resposta_ia.choices[0].message.content)

# =====================================================
# MENU PRINCIPAL
# =====================================================

if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print("  PROJETO DE RAG - EXERCÍCIOS PRÁTICOS  ")
        print("="*40)
        print("1 - Ler e Dividir Arquivo TXT")
        print("2 - Ler e Dividir Arquivo PDF")
        print("3 - Testar Geração de Embeddings")
        print("4 - Busca Semântica em Memória")
        print("5 - Geração de Resposta com RAG Simples")
        print("6 - Executar Pipeline RAG Completo (ChromaDB)")
        print("0 - Sair")
        print("="*40)

        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            exercicio_1_e_2_ler_txt()
        elif escolha == "2":
            exercicio_1_e_2_ler_pdf()
        elif escolha == "3":
            exercicio_3_gerar_embeddings()
        elif escolha == "4":
            exercicio_4_busca_semantica()
        elif escolha == "5":
            exercicio_5_rag_simples()
        elif escolha == "6":
            exercicio_6_pipeline_completo()
        elif escolha == "0":
            print("Encerrando o programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")