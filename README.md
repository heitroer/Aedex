# Aedex
Aedex: Sistema Inteligente de Previsão de Surtos de Dengue
O Aedex é uma solução de Machine Learning desenvolvida para transformar a gestão de arboviroses em Campo Grande/MS. Ao transitar de um modelo reativo para uma abordagem preditiva, o sistema permite que Unidades Básicas de Saúde (UBS) antecipem picos de demanda com 4 semanas de antecedência, otimizando estoques, alocação de pessoal e campanhas de prevenção.
🎯 O Problema
A gestão atual da dengue em ambientes de atenção primária é predominantemente reativa. O aumento no fluxo de atendimentos só é percebido após a instalação do surto, o que inviabiliza a resposta logística eficiente. Isso gera:
•	Desabastecimento: Escassez de insumos (testes, soros) em picos epidemiológicos.
•	Sobrecarga: Gargalos evitáveis nas UBS e UPAs.
•	Gestão ineficiente: Dificuldade em direcionar mutirões de endemias no tempo certo.
🚀 A Solução
O Aedex atua como um sistema de alerta precoce (Early Warning System). Utilizando modelos de Gradient Boosting, o sistema fornece uma projeção semanal precisa, categorizando o nível de risco (Verde a Vermelho) e fornecendo subsídios para que a Secretaria Municipal de Saúde tome decisões baseadas em dados, não em suposições.
📊 Fonte dos Dados
Utilizamos dados públicos e consolidados da plataforma InfoDengue (Fiocruz/FGV).
•	Abrangência: Séries temporais de 2016 a 2026.
•	Variáveis: Casos estimados, Rt (número de reprodução efetivo), indicadores de alerta e sazonalidade.
🛠 Metodologia
O projeto foi construído em Python, seguindo um pipeline robusto de ciência de dados:
	1.	Engenharia de Atributos: Criação de lags (1 a 8 semanas), médias móveis, transformações sazonais (seno/cosseno) e variáveis de tendência.
	2.	Modelagem: Implementação do XGBoost com regularização (L1/L2) para evitar overfitting.
	3.	Explicabilidade (XGB + SHAP): Implementação da biblioteca SHAP para garantir que as decisões do modelo sejam interpretáveis, permitindo entender o peso de cada variável na previsão final, eliminando a barreira da "caixa-preta".
	4.	Validação: Divisão temporal (walk-forward validation), treinando com dados históricos até 2022 e testando de 2023 em diante.
📈 Métricas de Desempenho
O modelo apresentou resultados de alta performance na validação:
•	Coeficiente de Determinação (‭$R^2$‬): 92,8% (na escala original).
•	Erro Percentual Absoluto Médio (MAPE): 26,8%.
•	Capacidade Preditiva: Identificação precisa de inflexões, picos sazonais e declínios com 4 semanas de antecedência.
⚠️ Limitações
Como qualquer modelo de IA, o Aedex possui limitações a serem consideradas:
•	Qualidade dos Dados: A acurácia depende da integridade dos registros semanais de notificações.
•	Fatores Exógenos: O modelo captura padrões autorregressivos, mas pode ser impactado por eventos atípicos (ex: mutações virais, mudanças climáticas extremas não previstas ou alterações severas no comportamento de mobilidade populacional).
💻 Como Rodar
O projeto foi desenvolvido para ser leve e acessível em qualquer ambiente.
	1.	Clone o repositório:
git clone https://github.com/seu-usuario/aedex.git
cd aedex

	2.	Instale as dependências:
pip install -r requirements.txt

	3.	Execute o Dashboard:
streamlit run app.py

🔗 Link do App
https://aedexdashboard.streamlit.app/
👥 Equipe
•	Heitor Leite de Oliveira Mesquita
•	Arthur Pereira Espindola
•	Emelly de Carvalho Koritiaki
•	Pâmela Rafaela do Prado
Projeto desenvolvido no Colégio Status, Campo Grande - MS | Ciências da Saúde - Saúde Coletiva.
Este sistema é um suporte à decisão e não substitui o julgamento clínico e a vigilância epidemiológica oficial.
