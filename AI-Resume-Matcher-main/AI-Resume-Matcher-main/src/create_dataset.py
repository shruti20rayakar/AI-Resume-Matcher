import pandas as pd

from config import PROCESSED_DATA_DIR


jobs = [
    {
        "job_id": "JOB001",
        "title": "Machine Learning Intern",
        "company": "TechNova",
        "description": (
            "We are looking for a Machine Learning Intern to work on "
            "machine learning models, data preprocessing, feature engineering, "
            "model evaluation and experimentation."
        ),
        "required_skills": (
            "Python, Machine Learning, Pandas, NumPy, Scikit-learn"
        ),
        "preferred_skills": (
            "TensorFlow, PyTorch, SQL, Git"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB002",
        "title": "Data Science Intern",
        "company": "DataWorks",
        "description": (
            "The intern will analyze datasets, build predictive models, "
            "perform exploratory data analysis and communicate analytical findings."
        ),
        "required_skills": (
            "Python, Pandas, NumPy, SQL, Statistics, Scikit-learn"
        ),
        "preferred_skills": (
            "Matplotlib, Seaborn, Machine Learning, Power BI"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB003",
        "title": "NLP Engineer Intern",
        "company": "LanguageAI",
        "description": (
            "Work on natural language processing systems involving text "
            "classification, text preprocessing, embeddings and Transformer models."
        ),
        "required_skills": (
            "Python, NLP, Transformers, Machine Learning"
        ),
        "preferred_skills": (
            "PyTorch, Hugging Face, BERT, Deep Learning"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB004",
        "title": "Deep Learning Intern",
        "company": "NeuralLabs",
        "description": (
            "Develop and evaluate deep learning models for real-world AI "
            "applications. Experience with neural networks and model training is expected."
        ),
        "required_skills": (
            "Python, Deep Learning, Neural Networks, Machine Learning"
        ),
        "preferred_skills": (
            "PyTorch, TensorFlow, CNN, Transformers"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB005",
        "title": "AI Engineer Intern",
        "company": "FutureMind",
        "description": (
            "Assist in developing AI applications using machine learning, "
            "deep learning and modern Transformer-based architectures."
        ),
        "required_skills": (
            "Python, Machine Learning, Deep Learning, Transformers"
        ),
        "preferred_skills": (
            "PyTorch, NLP, Computer Vision, Git"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB006",
        "title": "Computer Vision Intern",
        "company": "VisionTech",
        "description": (
            "Build computer vision solutions using image preprocessing, "
            "convolutional neural networks and deep learning."
        ),
        "required_skills": (
            "Python, Computer Vision, Deep Learning, OpenCV"
        ),
        "preferred_skills": (
            "PyTorch, TensorFlow, CNN, Object Detection"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB007",
        "title": "Data Analyst Intern",
        "company": "InsightCorp",
        "description": (
            "Analyze business data, create reports and dashboards, "
            "clean datasets and communicate insights to stakeholders."
        ),
        "required_skills": (
            "Python, SQL, Excel, Data Analysis"
        ),
        "preferred_skills": (
            "Pandas, Power BI, Tableau, Statistics"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB008",
        "title": "ML Research Intern",
        "company": "ResearchAI",
        "description": (
            "Explore machine learning and deep learning methods, conduct "
            "experiments and evaluate models on research datasets."
        ),
        "required_skills": (
            "Python, Machine Learning, Deep Learning, Statistics"
        ),
        "preferred_skills": (
            "PyTorch, Transformers, Research, Mathematics"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB009",
        "title": "Python Developer Intern",
        "company": "CodeSphere",
        "description": (
            "Develop Python applications, work with APIs and databases, "
            "write maintainable code and collaborate using version control."
        ),
        "required_skills": (
            "Python, Git, SQL, APIs"
        ),
        "preferred_skills": (
            "FastAPI, Flask, Docker, Linux"
        ),
        "experience_level": "Internship"
    },
    {
        "job_id": "JOB010",
        "title": "AI/NLP Research Intern",
        "company": "CognitiveSystems",
        "description": (
            "Research NLP problems involving text representation, semantic "
            "similarity, Transformer architectures and language understanding."
        ),
        "required_skills": (
            "Python, NLP, Transformers, Deep Learning"
        ),
        "preferred_skills": (
            "BERT, Hugging Face, PyTorch, Research"
        ),
        "experience_level": "Internship"
    }
]


df = pd.DataFrame(jobs)

output_path = PROCESSED_DATA_DIR / "jobs.csv"
df.to_csv(output_path, index=False)

print(f"Created dataset with {len(df)} job descriptions.")
print(f"Saved to: {output_path}")
print("\nColumns:")
print(list(df.columns))