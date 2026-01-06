# Arxiv Paper Recommendation System

A machine learning-based paper recommendation system that leverages TF-ICF (Term Frequency-Inverse Category Frequency) scoring with semantic analysis to suggest relevant research papers from the arXiv dataset.

## Overview

This project implements an intelligent paper recommendation engine that combines textual features (title + abstract), categorical information, and author network data to find similar papers. The system analyzes over 136,000 papers across 138 categories and provides personalized recommendations based on query text, categories, and author preferences.

## Features

### Multi-Modal Recommendation System
- **Content-based filtering** using TF-ICF weighted features
- **Semantic similarity** detection with cosine similarity metrics
- **Category-aware recommendations** - filter papers by research domain
- **Author-based filtering** - find papers from prolific authors
- **Hybrid approach** combining text, category, and author information

### Paper Matching Capabilities
- Find papers similar to a specific paper
- Search papers by natural language queries
- Filter by category (138 research categories supported)
- Filter by author(s) of interest
- Flexible combination of multiple filtering criteria

### Comprehensive Analysis
- Publication trend analysis across time periods
- Author productivity metrics and collaboration patterns
- Category evolution and growth trends
- Monthly publication seasonal patterns
- Average paper lag analysis

## Dataset

- **136,238** research papers from arXiv
- **138** research categories (Computer Science, Physics, Mathematics, etc.)
- **197,018** unique authors
- **15,000** features (after dimensionality reduction)
- **2,000** top authors tracked

## Technical Architecture

### Core Components

1. **Data Preprocessing**
   - Text cleaning (whitespace normalization, special character removal)
   - Author list parsing and structuring
   - Date parsing and temporal feature extraction

2. **Feature Engineering**
   - TF-ICF Features (15,000 dims) - Text content weighted by inverse category frequency
   - Category Embeddings (138 dims) - One-hot encoded research categories
   - Author Presence Matrix (2,000 dims) - Top 2,000 authors binary encoding

3. **Recommendation Algorithms**
   - **Similarity-Based**: Cosine similarity on combined feature space
   - **Filtering**: Category and author constraints applied post-similarity
   - **Ranking**: Top-k papers returned by similarity score

### Technology Stack

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (feature extraction, similarity metrics)
- **Visualization**: matplotlib, seaborn
- **Terminal UI**: rich (enhanced console output)

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/HimasagarU/Arxiv-Paper-Recommendation.git
cd Arxiv-Paper-Recommendation

# Install required packages
pip install pandas numpy scikit-learn scipy matplotlib seaborn rich
```

### Basic Usage Examples

#### 1. Find Similar Papers

```python
from notebook import recommend_similar_papers

# Get 5 papers similar to paper at index 10511
recommendations = recommend_similar_papers(
    paper_index=10511,
    feature_matrix=feature_matrix,
    df=df,
    top_n=5
)
```

#### 2. Query-Based Recommendations

```python
from notebook import recommend_papers_for_query

# Recommend papers matching a query
query = "neural networks for image classification"
recommendations = recommend_papers_for_query(
    query=query,
    vectorizer=count_vectorizer,
    feature_matrix=feature_matrix,
    df=df,
    top_n=5
)
```

#### 3. Filtered Recommendations

```python
# Search by category and authors
recommendations = recommend_papers_for_query(
    query="deep learning optimization",
    vectorizer=count_vectorizer,
    feature_matrix=feature_matrix,
    df=df,
    top_n=10,
    category="Artificial Intelligence",
    authors=["Yann LeCun", "Geoffrey Hinton"]
)
```

#### 4. Display Results

```python
from notebook import display_suggestions

# Format and display recommendations
display_suggestions(recommendations)
```

## Project Structure

```
Arxiv-Paper-Recommendation/
├── Notebook_009.ipynb              # Main analysis and recommendation engine
├── Report_009.pdf                  # Detailed technical report
├── arXiv_scientific dataset.csv     # Research paper dataset
├── README.md                        # This file
└── plots/                           # Generated analysis visualizations
    ├── total_papers_per_year.png
    ├── top_10_categories.png
    ├── top_10_authors.png
    ├── average_publication_lag.png
    ├── monthly_publication_counts.png
    ├── cumulative_papers_top_categories.png
    └── [other analysis plots]
```

## Key Insights

### Research Trends
- **Publication Growth**: Exponential increase in paper submissions over years
- **Category Leaders**: Artificial Intelligence and Machine Learning dominate research output
- **Collaboration**: Average authors per paper increasing (~4-5 co-authors trend)
- **Author Productivity**: Top authors publish 50-100+ papers each

### Recommendation Quality
- Cosine similarity scores range 0.98-0.99 for highly relevant papers
- Successfully groups papers across related sub-categories
- Category filtering improves recommendation precision

## Algorithm Details

### TF-ICF Weighting Scheme

```
TF-ICF(term_i, paper_j) = TF(term_i, paper_j) * ICF(term_i)
where:
  TF = Term count in paper
  ICF = log(Total Categories / Categories containing term)
```

This approach:
- Emphasizes rare but distinctive terms
- Reduces weight of common stop-words
- Captures category-specific terminology

### Similarity Computation

```
Similarity = Cosine(Paper1_Vector, Paper2_Vector)
           = (Vector1 · Vector2) / (||Vector1|| * ||Vector2||)
Range: [0, 1] (1 = identical, 0 = completely different)
```

## Results Example

**Query**: "We propose an improved dependency-directed backtracking method that reduces search progress loss"

**Top Recommendations**:
1. Dynamic Backtracking (Similarity: 0.6455)
2. Enhancing a Search Algorithm to Perform Intelligent Backtracking (Similarity: 0.5938)
3. Planning Graph as a Dynamic CSP (Similarity: 0.5362)

## Visualizations Generated

- **Time Series**: Publication counts by year, category trends
- **Heatmaps**: Category-year interaction matrix
- **Area Plots**: Cumulative paper counts by category
- **Bar Charts**: Top authors, top categories, growth metrics
- **Line Plots**: Author productivity trends, publication lag analysis

## Performance Characteristics

- **Preprocessing**: ~5-10 seconds for 136K papers
- **Feature Matrix**: 17,138 dimensions (sparse representation)
- **Recommendation Latency**: ~100ms per query
- **Memory**: ~2GB for feature matrix storage
- **Scalability**: Efficient sparse matrix operations support larger datasets

## Future Enhancements

- [ ] Deep learning embeddings (BERT/SciBERT)
- [ ] Temporal decay (recent papers weighted higher)
- [ ] Citation network integration
- [ ] Multi-language support
- [ ] Real-time paper ingestion pipeline
- [ ] Web interface for interactive exploration
- [ ] Graph neural networks for co-author networks
- [ ] Collaborative filtering component

## Limitations

- Static dataset (requires periodic updates)
- English language only
- No explicit user behavior feedback
- Category assignment based on arXiv metadata
- Limited to papers in the dataset

## Contributing

Contributions are welcome! Areas for improvement:
- Algorithm optimization
- New recommendation strategies
- UI/UX enhancements
- Dataset expansion
- Documentation improvements

## License

This project is available for educational and research purposes.

## Citation

If you use this recommendation system in your research, please cite:

```bibtex
@repository{arxiv_paper_recommendation,
  title={Arxiv Paper Recommendation System},
  author={Himasagar U},
  year={2025},
  url={https://github.com/HimasagarU/Arxiv-Paper-Recommendation}
}
```

## Contact & Support

For questions, issues, or suggestions, please open an issue on GitHub or contact the repository maintainer.

---

**Last Updated**: January 2025
**Notebook Version**: 009
**Python Version**: 3.x
