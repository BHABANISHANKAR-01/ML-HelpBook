// Machine Learning Learning Platform
class MLMasteryApp {
    constructor() {
        this.concepts = [];
        this.filteredConcepts = [];
        this.currentTopic = 'all';
        this.currentDifficulty = '';
        this.currentImplementation = '';
        this.currentLearningPath = 'all';
        this.searchQuery = '';
        this.studiedConcepts = new Set();
        this.bookmarkedConcepts = new Set();
        this.conceptNotes = new Map();
        this.isShowingBookmarks = false;
        this.currentConceptId = null;
        
        this.init();
    }

    async init() {
        this.loadConcepts();
        this.setupEventListeners();
        this.renderTopics();
        setTimeout(() => {
            this.renderConcepts();
            this.updateProgress();
            this.updateLearningPathInfo();
        }, 100);
    }

    loadConcepts() {
        const conceptsData = {
            "Prerequisites": [
                "Linear Algebra Fundamentals",
                "Vectors and Vector Operations",
                "Matrices and Matrix Operations",
                "Eigenvalues and Eigenvectors",
                "Singular Value Decomposition (SVD)",
                "Statistics and Probability",
                "Descriptive Statistics",
                "Probability Distributions",
                "Bayes Theorem",
                "Hypothesis Testing",
                "Central Limit Theorem",
                "Calculus for ML",
                "Derivatives and Partial Derivatives",
                "Chain Rule",
                "Gradient and Gradient Descent",
                "Optimization Fundamentals",
                "Python Programming",
                "NumPy for Numerical Computing",
                "Pandas for Data Manipulation",
                "Matplotlib and Seaborn for Visualization",
                "Jupyter Notebooks"
            ],
            "ML Fundamentals": [
                "Introduction to Machine Learning",
                "Types of Machine Learning",
                "Supervised Learning Overview",
                "Unsupervised Learning Overview", 
                "Reinforcement Learning Overview",
                "Semi-Supervised Learning",
                "ML Pipeline and Workflow",
                "Training, Validation, and Test Sets",
                "Bias-Variance Tradeoff",
                "Overfitting and Underfitting",
                "Cross-Validation Techniques",
                "Feature Engineering Fundamentals",
                "Data Collection and Quality",
                "Ethical AI and Fairness",
                "History and Evolution of ML"
            ],
            "Data Processing": [
                "Exploratory Data Analysis (EDA)",
                "Data Cleaning Techniques",
                "Handling Missing Data",
                "Outlier Detection and Treatment",
                "Data Visualization Best Practices",
                "Feature Selection Methods",
                "Feature Extraction Techniques",
                "Feature Scaling and Normalization",
                "Encoding Categorical Variables",
                "Handling Imbalanced Datasets",
                "Data Preprocessing Pipelines",
                "Time Series Data Processing",
                "Text Data Preprocessing",
                "Image Data Preprocessing",
                "Data Augmentation Techniques"
            ],
            "Supervised Learning": [
                "Linear Regression",
                "Multiple Linear Regression",
                "Polynomial Regression",
                "Regularization (Ridge, Lasso, Elastic Net)",
                "Logistic Regression",
                "Classification Metrics",
                "Decision Trees",
                "Random Forest",
                "Gradient Boosting Machines",
                "XGBoost and LightGBM",
                "Support Vector Machines (SVM)",
                "K-Nearest Neighbors (KNN)",
                "Naive Bayes Classifier",
                "Ensemble Methods",
                "Model Selection and Evaluation"
            ],
            "Unsupervised Learning": [
                "K-Means Clustering",
                "Hierarchical Clustering",
                "DBSCAN Clustering",
                "Gaussian Mixture Models",
                "Principal Component Analysis (PCA)",
                "Independent Component Analysis (ICA)",
                "t-SNE Dimensionality Reduction",
                "UMAP Dimensionality Reduction",
                "Anomaly Detection Methods",
                "Association Rule Learning",
                "Market Basket Analysis",
                "Clustering Evaluation Metrics",
                "Dimensionality Reduction Applications",
                "Feature Learning",
                "Self-Organizing Maps"
            ],
            "Deep Learning": [
                "Neural Network Fundamentals",
                "Perceptron and Multi-layer Perceptrons",
                "Activation Functions",
                "Backpropagation Algorithm",
                "Gradient Descent Variants",
                "Regularization in Deep Learning",
                "Convolutional Neural Networks (CNNs)",
                "CNN Architectures (LeNet, AlexNet, VGG, ResNet)",
                "Recurrent Neural Networks (RNNs)",
                "Long Short-Term Memory (LSTM)",
                "Gated Recurrent Units (GRU)",
                "Transformer Architecture",
                "Attention Mechanisms",
                "Generative Adversarial Networks (GANs)",
                "Autoencoders and Variational Autoencoders"
            ],
            "Advanced Topics": [
                "Natural Language Processing (NLP)",
                "Text Preprocessing and Tokenization",
                "Word Embeddings (Word2Vec, GloVe)",
                "Named Entity Recognition (NER)",
                "Sentiment Analysis",
                "Language Models and BERT",
                "Machine Translation",
                "Computer Vision Fundamentals",
                "Image Classification",
                "Object Detection (YOLO, R-CNN)",
                "Image Segmentation",
                "Face Recognition",
                "Style Transfer",
                "Reinforcement Learning",
                "Q-Learning and Policy Gradients"
            ],
            "Model Evaluation": [
                "Classification Metrics (Accuracy, Precision, Recall, F1)",
                "Regression Metrics (MSE, RMSE, MAE, R²)",
                "ROC Curves and AUC",
                "Precision-Recall Curves",
                "Cross-Validation Strategies",
                "Hyperparameter Tuning",
                "Grid Search and Random Search",
                "Bayesian Optimization",
                "Model Interpretability",
                "SHAP and LIME",
                "A/B Testing for ML",
                "Statistical Significance Testing",
                "Model Validation Techniques",
                "Learning Curves",
                "Model Comparison Methods"
            ],
            "Deployment & MLOps": [
                "Model Deployment Strategies",
                "REST APIs for ML Models",
                "Flask and FastAPI for Model Serving",
                "Containerization with Docker",
                "Cloud Deployment (AWS, GCP, Azure)",
                "Model Versioning",
                "Continuous Integration/Continuous Deployment (CI/CD)",
                "Model Monitoring and Maintenance",
                "Data Drift Detection",
                "Model Performance Monitoring",
                "MLOps Best Practices",
                "Feature Stores",
                "Model Registries",
                "Scalable ML Infrastructure",
                "Edge Deployment"
            ],
            "Real-World Projects": [
                "House Price Prediction (Regression)",
                "Customer Churn Prediction (Classification)", 
                "Recommendation System",
                "Sentiment Analysis of Social Media",
                "Image Classification with CNNs",
                "Time Series Forecasting",
                "Fraud Detection System",
                "Natural Language Chatbot",
                "Object Detection in Images",
                "Stock Price Prediction",
                "Healthcare Diagnosis Assistant",
                "E-commerce Product Recommendation",
                "Automated Text Summarization",
                "Predictive Maintenance",
                "Market Segmentation Analysis"
            ]
        };

        // Convert to concept objects
        let id = 1;
        Object.entries(conceptsData).forEach(([category, topics]) => {
            topics.forEach((title, index) => {
                this.concepts.push({
                    id: id++,
                    title: title,
                    category: category,
                    difficulty: this.getDifficulty(category, index),
                    description: this.generateDescription(title, category),
                    tags: this.generateTags(title, category),
                    learningPaths: this.getLearningPaths(category, index),
                    content: this.generateContent(title, category)
                });
            });
        });

        this.filteredConcepts = [...this.concepts];
    }

    getDifficulty(category, index) {
        const beginnerCategories = ['Prerequisites', 'ML Fundamentals'];
        const advancedCategories = ['Deep Learning', 'Advanced Topics', 'Deployment & MLOps'];
        
        if (beginnerCategories.includes(category)) {
            return index < 5 ? 'beginner' : 'intermediate';
        } else if (advancedCategories.includes(category)) {
            return index < 3 ? 'intermediate' : 'advanced';
        } else {
            return index < 5 ? 'beginner' : index < 10 ? 'intermediate' : 'advanced';
        }
    }

    getLearningPaths(category, index) {
        const paths = [];
        
        if (['Prerequisites', 'ML Fundamentals', 'Data Processing'].includes(category) && index < 10) {
            paths.push('beginner');
        }
        
        if (['Supervised Learning', 'Unsupervised Learning', 'Model Evaluation'].includes(category)) {
            paths.push('intermediate');
        }
        
        if (['Deep Learning', 'Advanced Topics', 'Deployment & MLOps'].includes(category)) {
            paths.push('advanced');
        }
        
        return paths.length > 0 ? paths : ['beginner'];
    }

    generateDescription(title, category) {
        const descriptions = {
            'Prerequisites': 'Master the fundamental mathematical and programming concepts essential for understanding machine learning algorithms.',
            'ML Fundamentals': 'Learn core machine learning concepts, terminology, and methodologies that form the foundation of all ML work.',
            'Data Processing': 'Develop skills in data preparation, cleaning, and feature engineering - crucial steps in any ML pipeline.',
            'Supervised Learning': 'Explore algorithms that learn from labeled training data to make predictions on new, unseen data.',
            'Unsupervised Learning': 'Discover patterns and structures in data without labeled examples using clustering and dimensionality reduction.',
            'Deep Learning': 'Dive into neural networks and deep learning architectures that have revolutionized AI applications.',
            'Advanced Topics': 'Explore specialized areas like NLP, computer vision, and reinforcement learning for complex real-world problems.',
            'Model Evaluation': 'Learn how to assess model performance, validate results, and ensure your models generalize well.',
            'Deployment & MLOps': 'Understand how to deploy models to production and maintain them with proper ML operations practices.',
            'Real-World Projects': 'Apply your knowledge to complete end-to-end projects that solve real business and research problems.'
        };
        
        return descriptions[category] || 'Learn essential machine learning concepts and techniques.';
    }

    generateTags(title, category) {
        const categoryTags = {
            'Prerequisites': ['math', 'programming', 'fundamentals'],
            'ML Fundamentals': ['theory', 'concepts', 'foundations'],
            'Data Processing': ['data', 'preprocessing', 'cleaning'],
            'Supervised Learning': ['algorithms', 'prediction', 'classification', 'regression'],
            'Unsupervised Learning': ['clustering', 'dimensionality reduction', 'pattern recognition'],
            'Deep Learning': ['neural networks', 'deep learning', 'AI'],
            'Advanced Topics': ['specialized', 'NLP', 'computer vision', 'reinforcement learning'],
            'Model Evaluation': ['metrics', 'validation', 'performance'],
            'Deployment & MLOps': ['production', 'deployment', 'operations'],
            'Real-World Projects': ['projects', 'applications', 'case studies']
        };
        
        let tags = [...(categoryTags[category] || [])];
        
        if (title.includes('Regression')) tags.push('regression');
        if (title.includes('Classification')) tags.push('classification');
        if (title.includes('Neural') || title.includes('CNN') || title.includes('RNN')) tags.push('neural networks');
        if (title.includes('Python')) tags.push('python');
        if (title.includes('Optimization')) tags.push('optimization');
        
        return tags.slice(0, 3);
    }

    generateContent(title, category) {
        return {
            overview: this.generateOverview(title, category),
            keyPoints: this.generateKeyPoints(title, category),
            mathFoundation: this.generateMathFoundation(title, category),
            implementations: this.generateImplementations(title, category),
            applications: this.generateApplications(title, category),
            resources: this.generateResources(title, category),
            relatedTopics: this.generateRelatedTopics(title, category)
        };
    }

    generateOverview(title, category) {
        const overviews = {
            'Linear Regression': 'Linear regression is a fundamental supervised learning algorithm that models the relationship between a dependent variable and independent variables by fitting a linear equation to observed data. It assumes a linear relationship and is used for prediction and understanding variable relationships.',
            'K-Means Clustering': 'K-Means is an unsupervised learning algorithm that partitions data into k clusters by minimizing within-cluster sum of squares. It iteratively assigns points to nearest centroids and updates centroids until convergence.',
            'Neural Network Fundamentals': 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information and learn patterns through training on data.',
            'Principal Component Analysis (PCA)': 'PCA is a dimensionality reduction technique that transforms data to lower dimensions while preserving maximum variance. It finds principal components that explain the most variance in the data.'
        };
        
        return overviews[title] || `${title} is an important concept in ${category.toLowerCase()} that provides essential knowledge and techniques for machine learning practitioners. Understanding this concept is crucial for building effective ML solutions.`;
    }

    generateKeyPoints(title, category) {
        const specificPoints = {
            'Linear Regression': [
                'Assumes linear relationship between features and target',
                'Uses least squares method to minimize prediction errors',
                'Provides interpretable coefficients for feature importance',
                'Sensitive to outliers and requires feature scaling',
                'Foundation for more complex regression techniques'
            ],
            'K-Means Clustering': [
                'Requires specifying number of clusters (k) beforehand',
                'Works best with spherical, similar-sized clusters',
                'Sensitive to initialization and outliers',
                'Uses Euclidean distance as similarity measure',
                'Computationally efficient for large datasets'
            ]
        };
        
        return specificPoints[title] || [
            'Understand the theoretical foundations and mathematical principles',
            'Learn when and why to apply this technique in real-world scenarios',
            'Master the implementation details and parameter tuning',
            'Recognize advantages, limitations, and potential pitfalls',
            'Apply best practices for optimal results'
        ];
    }

    generateMathFoundation(title, category) {
        const mathConcepts = {
            'Linear Regression': 'Cost function: J(θ) = 1/2m Σ(hθ(x) - y)². Gradient descent update: θ := θ - α∇J(θ). Normal equation: θ = (X^T X)^(-1) X^T y.',
            'K-Means Clustering': 'Objective function: J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||². Update centroids: μᵢ = 1/|Cᵢ| Σₓ∈Cᵢ x.',
            'Neural Network Fundamentals': 'Forward propagation: z = Wx + b, a = σ(z). Backpropagation: ∂C/∂w = ∂C/∂a · ∂a/∂z · ∂z/∂w.',
            'Gradient Descent': 'Update rule: θ := θ - α∇J(θ). Learning rate α controls step size. Convergence when ||∇J(θ)|| < ε.'
        };
        
        return mathConcepts[title] || 'Mathematical foundations involve statistical and linear algebra concepts that underpin the algorithm\'s behavior and performance.';
    }

    generateImplementations(title, category) {
        const pythonCode = `# ${title} Implementation in Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example implementation for ${title}
def ${title.toLowerCase().replace(/[^a-z0-9]/g, '_')}_example():
    """
    Comprehensive example of ${title}
    """
    # Generate or load sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training ${title}...")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Run example
result = ${title.toLowerCase().replace(/[^a-z0-9]/g, '_')}_example()
print("Implementation completed successfully!")`;

        const rCode = `# ${title} Implementation in R
library(tidyverse)
library(caret)
library(ggplot2)

# ${title} example implementation
${title.toLowerCase().replace(/[^a-z0-9]/g, '.')}.example <- function() {
  # Generate sample data
  set.seed(42)
  n <- 100
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  y <- X1 + 2 * X2 + rnorm(n, sd = 0.1)
  
  data <- data.frame(X1 = X1, X2 = X2, y = y)
  
  # Split data
  trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]
  
  cat("Training ${title}...\\n")
  cat(paste("Training set size:", nrow(trainData), "\\n"))
  cat(paste("Test set size:", nrow(testData), "\\n"))
  
  return(list(train = trainData, test = testData))
}

# Run example
result <- ${title.toLowerCase().replace(/[^a-z0-9]/g, '.')}.example()
cat("Implementation completed successfully!\\n")`;

        const jsCode = `// ${title} Implementation in JavaScript
class ${title.replace(/[^A-Za-z0-9]/g, '')} {
  constructor(options = {}) {
    this.options = {
      learningRate: 0.01,
      maxIterations: 1000,
      tolerance: 1e-6,
      ...options
    };
    this.trained = false;
  }
  
  /**
   * Train the ${title} model
   * @param {Array} X - Feature matrix
   * @param {Array} y - Target values
   */
  fit(X, y) {
    console.log('Training ${title}...');
    console.log(\`Training data shape: \${X.length} x \${X[0].length}\`);
    
    // Implementation would go here
    this.trained = true;
    
    console.log('Training completed!');
    return this;
  }
  
  /**
   * Make predictions
   * @param {Array} X - Feature matrix
   * @returns {Array} Predictions
   */
  predict(X) {
    if (!this.trained) {
      throw new Error('Model must be trained before making predictions');
    }
    
    console.log('Making predictions...');
    // Implementation would go here
    return new Array(X.length).fill(0);
  }
  
  /**
   * Get model parameters
   * @returns {Object} Model parameters
   */
  getParams() {
    return this.options;
  }
}

// Example usage
const model = new ${title.replace(/[^A-Za-z0-9]/g, '')}();
// const X = [[1, 2], [3, 4], [5, 6]];
// const y = [1, 2, 3];
// model.fit(X, y);
// const predictions = model.predict(X);

console.log('${title} implementation ready');`;

        return {
            python: pythonCode,
            r: rCode,
            javascript: jsCode
        };
    }

    generateApplications(title, category) {
        return [
            {
                title: 'Business Analytics',
                description: 'Applied in customer segmentation, demand forecasting, and market analysis to drive business decisions.'
            },
            {
                title: 'Healthcare',
                description: 'Used for disease diagnosis, drug discovery, and personalized treatment recommendations.'
            },
            {
                title: 'Technology',
                description: 'Implemented in recommendation systems, image recognition, and natural language processing applications.'
            },
            {
                title: 'Finance',
                description: 'Utilized for fraud detection, risk assessment, algorithmic trading, and credit scoring.'
            }
        ];
    }

    generateResources(title, category) {
        return [
            { title: 'Scikit-learn Documentation', url: 'https://scikit-learn.org/stable/' },
            { title: 'Machine Learning Course', url: 'https://www.coursera.org/learn/machine-learning' },
            { title: 'Hands-On ML Book', url: 'https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/' },
            { title: 'Kaggle Learn', url: 'https://www.kaggle.com/learn' },
            { title: 'ML Papers', url: 'https://arxiv.org/list/stat.ML/recent' }
        ];
    }

    generateRelatedTopics(title, category) {
        const related = {
            'Linear Regression': ['Multiple Linear Regression', 'Polynomial Regression', 'Regularization (Ridge, Lasso, Elastic Net)'],
            'K-Means Clustering': ['Hierarchical Clustering', 'DBSCAN Clustering', 'Gaussian Mixture Models'],
            'Neural Network Fundamentals': ['Perceptron and Multi-layer Perceptrons', 'Activation Functions', 'Backpropagation Algorithm']
        };
        
        return related[title] || ['Feature Engineering Fundamentals', 'Model Evaluation', 'Cross-Validation Techniques'];
    }

    setupEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchQuery = e.target.value.toLowerCase().trim();
                this.filterConcepts();
            });
        }

        // Filters
        const difficultyFilter = document.getElementById('difficultyFilter');
        if (difficultyFilter) {
            difficultyFilter.addEventListener('change', (e) => {
                this.currentDifficulty = e.target.value;
                this.filterConcepts();
            });
        }

        const implementationFilter = document.getElementById('implementationFilter');
        if (implementationFilter) {
            implementationFilter.addEventListener('change', (e) => {
                this.currentImplementation = e.target.value;
                this.filterConcepts();
            });
        }

        // Learning path selector
        const learningPath = document.getElementById('learningPath');
        if (learningPath) {
            learningPath.addEventListener('change', (e) => {
                this.currentLearningPath = e.target.value;
                this.filterConcepts();
                this.updateLearningPathInfo();
            });
        }

        // Clear filters
        const clearFiltersBtn = document.getElementById('clearFilters');
        if (clearFiltersBtn) {
            clearFiltersBtn.addEventListener('click', () => {
                this.clearFilters();
            });
        }

        // Bookmarks toggle
        const bookmarksToggle = document.getElementById('bookmarksToggle');
        if (bookmarksToggle) {
            bookmarksToggle.addEventListener('click', () => {
                this.toggleBookmarks();
            });
        }

        // Mobile menu
        const mobileMenuToggle = document.getElementById('mobileMenuToggle');
        const sidebar = document.getElementById('sidebar');
        if (mobileMenuToggle && sidebar) {
            mobileMenuToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
        }

        // Modal functionality
        this.setupModalListeners();

        // Notes functionality
        this.setupNotesListeners();

        // Code tabs
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab-btn')) {
                this.switchCodeTab(e.target.dataset.lang);
            }
        });

        // Copy code functionality
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-btn')) {
                this.copyCode(e.target);
            }
        });
    }

    setupModalListeners() {
        const modalClose = document.getElementById('modalClose');
        const modalBackdrop = document.getElementById('modalBackdrop');
        
        if (modalClose) {
            modalClose.addEventListener('click', () => this.closeModal());
        }
        if (modalBackdrop) {
            modalBackdrop.addEventListener('click', () => this.closeModal());
        }

        // Bookmark and notes buttons
        const bookmarkBtn = document.getElementById('bookmarkBtn');
        const notesBtn = document.getElementById('notesBtn');
        
        if (bookmarkBtn) {
            bookmarkBtn.addEventListener('click', () => this.toggleBookmark());
        }
        if (notesBtn) {
            notesBtn.addEventListener('click', () => this.openNotesModal());
        }

        // Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
                this.closeNotesModal();
            }
        });
    }

    setupNotesListeners() {
        const notesClose = document.getElementById('notesClose');
        const notesBackdrop = document.getElementById('notesBackdrop');
        const saveNotes = document.getElementById('saveNotes');
        const clearNotes = document.getElementById('clearNotes');

        if (notesClose) {
            notesClose.addEventListener('click', () => this.closeNotesModal());
        }
        if (notesBackdrop) {
            notesBackdrop.addEventListener('click', () => this.closeNotesModal());
        }
        if (saveNotes) {
            saveNotes.addEventListener('click', () => this.saveNotes());
        }
        if (clearNotes) {
            clearNotes.addEventListener('click', () => this.clearConceptNotes());
        }
    }

    renderTopics() {
        const topicList = document.getElementById('topicList');
        if (!topicList) return;

        const categories = [...new Set(this.concepts.map(c => c.category))];
        
        topicList.innerHTML = `
            <li class="topic-item">
                <div class="topic-link active" data-topic="all">
                    <span>All Concepts</span>
                    <span class="topic-count">${this.concepts.length}</span>
                </div>
            </li>
        `;

        categories.forEach(category => {
            const count = this.concepts.filter(c => c.category === category).length;
            const li = document.createElement('li');
            li.className = 'topic-item';
            li.innerHTML = `
                <div class="topic-link" data-topic="${category}">
                    <span>${category}</span>
                    <span class="topic-count">${count}</span>
                </div>
            `;
            topicList.appendChild(li);
        });

        // Add click listeners
        topicList.addEventListener('click', (e) => {
            const topicLink = e.target.closest('.topic-link');
            if (topicLink) {
                e.preventDefault();
                const topic = topicLink.dataset.topic;
                this.selectTopic(topic);
            }
        });
    }

    selectTopic(topic) {
        this.currentTopic = topic;
        
        // Update active state
        document.querySelectorAll('.topic-link').forEach(link => {
            link.classList.remove('active');
        });
        const activeLink = document.querySelector(`[data-topic="${topic}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
        
        // Update header
        const currentTopicElement = document.getElementById('currentTopic');
        const breadcrumb = document.getElementById('breadcrumb');
        if (currentTopicElement) {
            currentTopicElement.textContent = topic === 'all' ? 'All ML Concepts' : topic;
        }
        if (breadcrumb) {
            breadcrumb.innerHTML = topic === 'all' ? '<span>Home</span>' : `<span>Home</span><span>${topic}</span>`;
        }
        
        this.filterConcepts();
        
        // Close mobile menu
        const sidebar = document.getElementById('sidebar');
        if (sidebar) {
            sidebar.classList.remove('open');
        }
    }

    filterConcepts() {
        let filtered = [...this.concepts];

        // Filter by topic
        if (this.currentTopic !== 'all') {
            filtered = filtered.filter(c => c.category === this.currentTopic);
        }

        // Filter by difficulty
        if (this.currentDifficulty) {
            filtered = filtered.filter(c => c.difficulty === this.currentDifficulty);
        }

        // Filter by learning path
        if (this.currentLearningPath !== 'all') {
            filtered = filtered.filter(c => c.learningPaths.includes(this.currentLearningPath));
        }

        // Filter by search query
        if (this.searchQuery) {
            filtered = filtered.filter(c => 
                c.title.toLowerCase().includes(this.searchQuery) ||
                c.category.toLowerCase().includes(this.searchQuery) ||
                c.tags.some(tag => tag.toLowerCase().includes(this.searchQuery))
            );
        }

        // Filter by bookmarks
        if (this.isShowingBookmarks) {
            filtered = filtered.filter(c => this.bookmarkedConcepts.has(c.id));
        }

        this.filteredConcepts = filtered;
        this.renderConcepts();
    }

    renderConcepts() {
        const conceptsGrid = document.getElementById('conceptsGrid');
        const loading = document.getElementById('loading');
        const noResults = document.getElementById('noResults');

        if (!conceptsGrid || !loading || !noResults) return;

        loading.classList.remove('hidden');
        noResults.classList.add('hidden');
        conceptsGrid.innerHTML = '';

        setTimeout(() => {
            loading.classList.add('hidden');

            if (this.filteredConcepts.length === 0) {
                noResults.classList.remove('hidden');
                return;
            }

            const conceptsHTML = this.filteredConcepts.map(concept => `
                <div class="concept-card ${this.studiedConcepts.has(concept.id) ? 'completed' : ''} ${this.bookmarkedConcepts.has(concept.id) ? 'bookmarked' : ''}" 
                     data-id="${concept.id}">
                    <div class="concept-meta">
                        <span class="status status--${concept.difficulty}">${concept.difficulty}</span>
                        <span class="concept-category">${concept.category}</span>
                    </div>
                    <h3 class="concept-title">${concept.title}</h3>
                    <p class="concept-description">${concept.description}</p>
                    <div class="concept-tags">
                        ${concept.tags.map(tag => `<span class="concept-tag">${tag}</span>`).join('')}
                    </div>
                </div>
            `).join('');

            conceptsGrid.innerHTML = conceptsHTML;

            // Update concept count
            const conceptCountElement = document.getElementById('conceptCount');
            if (conceptCountElement) {
                conceptCountElement.textContent = 
                    `${this.filteredConcepts.length} concept${this.filteredConcepts.length === 1 ? '' : 's'}`;
            }

            this.setupConceptCardListeners();
        }, 200);
    }

    setupConceptCardListeners() {
        const conceptsGrid = document.getElementById('conceptsGrid');
        if (!conceptsGrid) return;

        // Remove existing listener if any
        if (this.conceptCardClickHandler) {
            conceptsGrid.removeEventListener('click', this.conceptCardClickHandler);
        }
        
        this.conceptCardClickHandler = (e) => {
            const conceptCard = e.target.closest('.concept-card');
            if (conceptCard) {
                e.preventDefault();
                e.stopPropagation();
                const conceptId = parseInt(conceptCard.dataset.id);
                this.openConceptModal(conceptId);
            }
        };

        conceptsGrid.addEventListener('click', this.conceptCardClickHandler);
    }

    openConceptModal(conceptId) {
        const concept = this.concepts.find(c => c.id === conceptId);
        if (!concept) return;

        this.currentConceptId = conceptId;
        this.studiedConcepts.add(conceptId);
        this.updateProgress();

        // Populate modal
        this.populateModal(concept);

        // Show modal
        const modal = document.getElementById('conceptModal');
        if (modal) {
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }

        // Highlight code
        setTimeout(() => {
            if (window.Prism) {
                Prism.highlightAll();
            }
        }, 100);

        this.renderConcepts();
    }

    populateModal(concept) {
        // Header
        const modalTitle = document.getElementById('modalTitle');
        const modalDifficulty = document.getElementById('modalDifficulty');
        const modalCategory = document.getElementById('modalCategory');

        if (modalTitle) modalTitle.textContent = concept.title;
        if (modalDifficulty) {
            modalDifficulty.className = `status status--${concept.difficulty}`;
            modalDifficulty.textContent = concept.difficulty;
        }
        if (modalCategory) modalCategory.textContent = concept.category;

        // Bookmark button
        const bookmarkBtn = document.getElementById('bookmarkBtn');
        if (bookmarkBtn) {
            bookmarkBtn.classList.toggle('active', this.bookmarkedConcepts.has(concept.id));
        }

        // Notes button
        const notesBtn = document.getElementById('notesBtn');
        if (notesBtn) {
            notesBtn.classList.toggle('active', this.conceptNotes.has(concept.id));
        }

        // Content sections
        this.populateContentSections(concept);
    }

    populateContentSections(concept) {
        const { content } = concept;

        // Overview
        const overviewElement = document.getElementById('conceptOverview');
        if (overviewElement) {
            overviewElement.innerHTML = `<h4>Overview</h4><p>${content.overview}</p>`;
        }

        // Key points
        const keyPointsElement = document.getElementById('keyPoints');
        if (keyPointsElement) {
            keyPointsElement.innerHTML = `
                <h4>Key Points</h4>
                <ul>${content.keyPoints.map(point => `<li>${point}</li>`).join('')}</ul>
            `;
        }

        // Math foundation
        const mathFoundationElement = document.getElementById('mathFoundation');
        if (mathFoundationElement) {
            mathFoundationElement.innerHTML = `
                <h4>Mathematical Foundation</h4>
                <p>${content.mathFoundation}</p>
            `;
        }

        // Code implementations
        this.populateCodeSections(content.implementations);

        // Applications
        const applicationsElement = document.getElementById('applicationsContent');
        if (applicationsElement) {
            applicationsElement.innerHTML = content.applications.map(app => `
                <div class="application-item">
                    <h5>${app.title}</h5>
                    <p>${app.description}</p>
                </div>
            `).join('');
        }

        // Resources
        const resourcesElement = document.getElementById('resourcesContent');
        if (resourcesElement) {
            resourcesElement.innerHTML = `
                <div class="resource-links">
                    ${content.resources.map(resource => `
                        <a href="${resource.url}" target="_blank" class="resource-link">${resource.title}</a>
                    `).join('')}
                </div>
            `;
        }

        // Related topics
        const relatedTopicsElement = document.getElementById('relatedTopics');
        if (relatedTopicsElement) {
            relatedTopicsElement.innerHTML = content.relatedTopics.map(topic => `
                <span class="related-topic">${topic}</span>
            `).join('');
        }
    }

    populateCodeSections(implementations) {
        const languages = ['python', 'r', 'javascript'];
        
        languages.forEach(lang => {
            const element = document.getElementById(`${lang}Code`);
            if (element) {
                element.innerHTML = `
                    <div class="code-block">
                        <div class="code-header">
                            <span>${lang.charAt(0).toUpperCase() + lang.slice(1)} Implementation</span>
                            <button class="btn btn--sm copy-btn">Copy</button>
                        </div>
                        <pre><code class="language-${lang === 'javascript' ? 'js' : lang}">${implementations[lang]}</code></pre>
                    </div>
                `;
            }
        });
    }

    switchCodeTab(language) {
        // Update active tab
        document.querySelectorAll('.tab-btn').forEach(tab => {
            tab.classList.remove('active');
        });
        const activeTab = document.querySelector(`[data-lang="${language}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
        }

        // Show corresponding content
        document.querySelectorAll('.code-section').forEach(section => {
            section.classList.remove('active');
        });
        const activeSection = document.querySelector(`.code-section[data-lang="${language}"]`);
        if (activeSection) {
            activeSection.classList.add('active');
        }
    }

    closeModal() {
        const modal = document.getElementById('conceptModal');
        if (modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = '';
        }
        this.currentConceptId = null;
    }

    toggleBookmark() {
        if (!this.currentConceptId) return;

        const bookmarkBtn = document.getElementById('bookmarkBtn');
        if (!bookmarkBtn) return;

        if (this.bookmarkedConcepts.has(this.currentConceptId)) {
            this.bookmarkedConcepts.delete(this.currentConceptId);
            bookmarkBtn.classList.remove('active');
        } else {
            this.bookmarkedConcepts.add(this.currentConceptId);
            bookmarkBtn.classList.add('active');
        }

        if (this.isShowingBookmarks) {
            this.filterConcepts();
        } else {
            this.renderConcepts();
        }
    }

    toggleBookmarks() {
        const bookmarksBtn = document.getElementById('bookmarksToggle');
        if (!bookmarksBtn) return;

        this.isShowingBookmarks = !this.isShowingBookmarks;

        const spanElement = bookmarksBtn.querySelector('span');
        if (this.isShowingBookmarks) {
            if (spanElement) spanElement.textContent = 'Show All';
            bookmarksBtn.classList.add('btn--primary');
            bookmarksBtn.classList.remove('btn--outline');
        } else {
            if (spanElement) spanElement.textContent = 'Show Bookmarks';
            bookmarksBtn.classList.remove('btn--primary');
            bookmarksBtn.classList.add('btn--outline');
        }

        this.filterConcepts();
    }

    openNotesModal() {
        if (!this.currentConceptId) return;

        const notesModal = document.getElementById('notesModal');
        const notesTextarea = document.getElementById('notesTextarea');
        
        if (notesModal && notesTextarea) {
            notesTextarea.value = this.conceptNotes.get(this.currentConceptId) || '';
            notesModal.classList.remove('hidden');
        }
    }

    closeNotesModal() {
        const notesModal = document.getElementById('notesModal');
        if (notesModal) {
            notesModal.classList.add('hidden');
        }
    }

    saveNotes() {
        if (!this.currentConceptId) return;

        const notesTextarea = document.getElementById('notesTextarea');
        const notesBtn = document.getElementById('notesBtn');
        
        if (notesTextarea && notesBtn) {
            const notes = notesTextarea.value.trim();
            if (notes) {
                this.conceptNotes.set(this.currentConceptId, notes);
                notesBtn.classList.add('active');
            } else {
                this.conceptNotes.delete(this.currentConceptId);
                notesBtn.classList.remove('active');
            }
        }
        
        this.closeNotesModal();
    }

    clearConceptNotes() {
        const notesTextarea = document.getElementById('notesTextarea');
        if (notesTextarea) {
            notesTextarea.value = '';
        }
    }

    clearFilters() {
        const searchInput = document.getElementById('searchInput');
        const difficultyFilter = document.getElementById('difficultyFilter');
        const implementationFilter = document.getElementById('implementationFilter');

        if (searchInput) searchInput.value = '';
        if (difficultyFilter) difficultyFilter.value = '';
        if (implementationFilter) implementationFilter.value = '';

        this.searchQuery = '';
        this.currentDifficulty = '';
        this.currentImplementation = '';
        this.filterConcepts();
    }

    copyCode(button) {
        const codeBlock = button.closest('.code-block');
        if (!codeBlock) return;

        const code = codeBlock.querySelector('code');
        if (!code) return;

        const text = code.textContent;

        navigator.clipboard.writeText(text).then(() => {
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            button.classList.add('btn--primary');
            
            setTimeout(() => {
                button.textContent = originalText;
                button.classList.remove('btn--primary');
            }, 1500);
        }).catch(err => {
            console.error('Failed to copy: ', err);
        });
    }

    updateProgress() {
        const progressText = document.getElementById('progressText');
        const progressCircle = document.querySelector('.progress-circle');
        
        if (!progressText || !progressCircle) return;

        const studied = this.studiedConcepts.size;
        const total = this.concepts.length;
        const percentage = total > 0 ? (studied / total) * 100 : 0;

        progressText.textContent = `${studied}/${total}`;
        progressCircle.style.background = `conic-gradient(var(--color-primary) ${percentage * 3.6}deg, var(--color-secondary) 0deg)`;
    }

    updateLearningPathInfo() {
        const pathProgress = document.getElementById('pathProgress');
        const pathEstimate = document.getElementById('pathEstimate');
        
        if (!pathProgress || !pathEstimate) return;

        if (this.currentLearningPath === 'all') {
            pathProgress.textContent = '0%';
            pathEstimate.textContent = '--';
            return;
        }

        const pathConcepts = this.concepts.filter(c => c.learningPaths.includes(this.currentLearningPath));
        const studiedInPath = pathConcepts.filter(c => this.studiedConcepts.has(c.id)).length;
        const progressPercentage = pathConcepts.length > 0 ? Math.round((studiedInPath / pathConcepts.length) * 100) : 0;

        const timeEstimates = {
            'beginner': '4-6 weeks',
            'intermediate': '6-8 weeks',
            'advanced': '8-12 weeks'
        };

        pathProgress.textContent = `${progressPercentage}%`;
        pathEstimate.textContent = timeEstimates[this.currentLearningPath] || '--';
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    new MLMasteryApp();
});