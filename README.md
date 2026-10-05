# Foco Técnico: Álgebra Linear, Machine Learning & Privacidade de Dados

Este repositório contém o desenvolvimento de um projeto prático focado na aplicação de **Álgebra Linear** em tarefas de **Machine Learning** e **Proteção de Dados** no setor de seguros. 

O objetivo principal foi resolver desafios práticos de classificação e regressão, demonstrando analítica e computacionalmente o impacto do escalonamento de atributos e métodos de ofuscação de dados para preservar a privacidade dos clientes.

---

## 📌 Destaques do Projetos

1. **Impacto do Escalonamento no Algoritmo kNN:**
   - Demonstração prática da sensibilidade do algoritmo k-Nearest Neighbors à escala das variáveis.
   - O uso do `StandardScaler` permitiu que o $F_1$ Score do classificador saltasse de **~0.12 (dados brutos)** para **0.92 (dados escalados)**, superando significativamente inclusive a baseline do modelo aleatório ($F_1 \approx 0.21$).

2. **Invariância da Regressão Linear à Escala:**
   - Comprovação de que o escalonamento dos atributos altera a escala dos pesos ($w$) do modelo, mas **preserva exatamente as predições finais ($\hat{y}$)**, mantendo as métricas $R^2$ e REQM constantes.

3. **Ofuscação de Dados e Privacidade (Transformação Matricial):**
   - Implementação de mascaramento de dados sensíveis multiplicando a matriz de características $X$ por uma matriz quadrada invertível aleatória $P$ ($X \cdot P$).
   - Prova matemática e computacional de que a ofuscação não altera os resultados preditivos do modelo nem as métricas de avaliação, permitindo treinar modelos com dados protegidos.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

- **Linguagem:** Python
- **Manipulação de Dados:** Pandas, NumPy
- **Machine Learning & Pré-processamento:** Scikit-Learn (`KNeighborsClassifier`, `StandardScaler`, `f1_score`, `mean_squared_error`, `r2_score`)
- **Álgebra Linear:** Operações matriciais, matriz inversa e determinantes com NumPy (`np.linalg`)
- **Ambiente:** Jupyter Notebook / Visual Studio Code

---

## 🏁 Conclusões

O projeto comprovou que a Álgebra Linear é a base indispensável para a construção, otimização e segurança de modelos de aprendizado de máquina. A transformação matricial aplicada permitiu garantir a conformidade com a privacidade dos dados de clientes sem comprometer o desempenho dos modelos analíticos.
