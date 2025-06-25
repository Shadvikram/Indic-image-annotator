#  Contributing to AI-Powered Indic Image Annotator

We welcome contributions from the community! Whether you're a beginner or an expert, your help improves the tool for everyone working with Indic languages.

---

##  Before You Start

- Check the [Issues](https://github.com/your-repo/issues) for open tasks
- Follow the coding and formatting standards mentioned below
- All contributions must go through a pull request (PR)

---

##  How to Contribute

### 1. Fork and Clone the Repo

```bash
git clone https://github.com/your-username/indic-image-annotator.git
cd indic-image-annotator
```

### 2. Create a New Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Add new features, fix bugs, or improve documentation
- Keep commits small and meaningful

### 4. Test Your Changes

Run the app locally:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest
```

---

##  Development Setup

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run type checks:

```bash
mypy app.py utils.py config.py
```

Format code:

```bash
black app.py utils.py config.py
```

---

##  Code Style Guidelines

- Use `black` for formatting
- Follow PEP8 conventions
- Add comments and docstrings where needed
- Keep functions modular and reusable

---

##  Pull Request Checklist

Before creating a pull request:

- [ ] Is your code linted (`black`)?
- [ ] Did you test your feature thoroughly?
- [ ] Does your feature break anything else?
- [ ] Are docstrings and inline comments added?
- [ ] Is your branch up to date with `main`?

---

##  Suggestions for Contribution

- Improve language translations or add new ones
- Optimize model performance or loading
- Add support for new export formats
- Build UI enhancements or accessibility improvements
- Write additional unit tests
- Improve documentation or tutorials
- Add support for right-to-left (RTL) scripts

---

##  Code of Conduct

We expect all contributors to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, inclusive, and open in all interactions.

---

##  Project Roadmap

- [ ] Video annotation support
- [ ] Custom model integration
- [ ] Mobile-friendly UI
- [ ] Collaborative annotation features

---

##  Acknowledgment

Thank you for being a part of the Indic AI community and helping grow open-source tools that support our languages!

Made with ❤️ for the Indic language ecosystem.