text = """
JOSEPH MARGARYAN
Email: josephmargaryan@gmail.com | GitHub: github.com/josephmargaryan | LinkedIn: linkedin.com/in/joseph-margaryan-256961222/

RESEARCH PROFILE (data science)
• MSc Bioinformatics (Computer Science), University of Copenhagen (expected MM YYYY). Focus: probabilistic ML, score-based diffusion, Bayesian conditioning under soft evidence.
• Built an end-to-end SMPK prototype (PK-guided diffusion) in JAX/Equinox: DSM-trained reference score in observable space + guided sampling; validated on synthetic 2D tasks and an MNIST proof-of-concept.
• Selected grades (12/12): Advanced Topics in Deep Learning; Advanced Topics in Machine Learning; Machine Learning B; [add 1–3 more max].
• 1 year part-time industry ML (Base Life Science): production codebase, reproducible experiments, evaluation/failure analysis.

EDUCATION
2022–present  University of Copenhagen — MSc Bioinformatics (CS specialization), expected MM YYYY.
MSc thesis (planned): differential privacy for medical data settings; parallel research in probabilistic generative modeling.
2019–2022      University of Copenhagen — BSc Health Informatics (thesis grade: 12/12).
Thesis: tumor segmentation/classification with Rigshospitalet; rigorous validation and reproducibility.

RESEARCH EXPERIENCE
2025–present  Research project (with [PI/group], UCPH): Score-Matched Probability Kinematics for diffusion models.
• Derived/implemented PK-guided posterior score for soft evidence p(d) over observables d=T(x).
• Implemented diffusion training + sampling in JAX/Equinox (EDM objective; samplers: [name the ones you actually used]).
• Trained reference score network sπ(d) via DSM on d=T(x) samples; empirically validated guidance stability.

2024–2025  Bayesian ML / spectral methods project (with [PI/group]).
• Implemented probabilistic/spectral components and ran controlled experiments; manuscript status: [in preparation / under review], title: [..].

PUBLICATIONS / PREPRINTS
• [Only include real entries. If none, remove this section.]

INDUSTRY EXPERIENCE
2024–2025  Base Life Science — Student Data Scientist (part-time).
• Implemented long-sequence document classification (Hugging Face + PyTorch + scikit-learn); evaluation and failure-mode analysis.
• Contributed to production workflows: code reviews, integration, maintainable configs/READMEs, stakeholder collaboration.

TEACHING
2024–present  Teaching Assistant, Python Programming for Data Science (MSc), UCPH (2 runs): exercises, debugging, grading.
2025–present  Teaching Assistant, Foundations of Data Science (BSc), UCPH.

SKILLS
Python, JAX/Equinox, PyTorch; NumPy/pandas; scikit-learn; Hugging Face.
Diffusion/score matching, Bayesian ML, uncertainty/evaluation, computer vision/medical imaging.
Git, reproducibility, experiment organization, documentation.

Languages: Danish (fluent), English (fluent).
"""
cv = text
print(len(cv))
