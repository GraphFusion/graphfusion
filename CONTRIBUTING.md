# Contributing to GraphFusionAI

Thank you for considering contributing to **GraphFusionAI**! We’re thrilled to have you here, and your support will help us improve this evolving library. This document outlines how you can contribute effectively.

---

## 📋 **Table of Contents**

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Reporting Issues](#reporting-issues)
4. [Submitting Pull Requests](#submitting-pull-requests)
5. [Coding Guidelines](#coding-guidelines)
6. [Testing and Documentation](#testing-and-documentation)
7. [Community](#community)

---

## 🚦 **Code of Conduct**
We are committed to fostering a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

## 🛠️ **How to Contribute**
There are several ways you can help improve **GraphFusionAI**:

1. **Report bugs**: Submit clear and concise bug reports.
2. **Fix issues**: Browse our [GitHub Issues](https://github.com/yourusername/graphfusionai/issues) and find one to fix.
3. **Add features**: Propose and implement new features.
4. **Improve documentation**: Fix typos, clarify concepts, or add examples.
5. **Write tests**: Improve test coverage for core modules.

---

## 🐞 **Reporting Issues**

If you find a bug or experience unexpected behavior, please submit an issue:

1. Go to the **Issues** tab in our repository.
2. Click on `New Issue` and choose the relevant template.
3. Provide the following information:
   - **Description**: A clear explanation of the issue.
   - **Steps to Reproduce**: A step-by-step guide to recreate the issue.
   - **Expected Behavior**: What you expected to happen.
   - **Actual Behavior**: What actually happened.
   - **Environment**: Python version, OS, and library version.
4. Include screenshots or logs if applicable.

---

## 🚀 **Submitting Pull Requests**

To contribute code or improvements:

1. **Fork the repository**:
   - Go to the repository and click `Fork`.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/graphfusionai.git
   cd graphfusionai
   ```
3. **Create a branch**:
   Use a descriptive branch name:
   ```bash
   git checkout -b fix-issue-123
   ```
4. **Make changes**:
   - Follow the [Coding Guidelines](#coding-guidelines).
   - Run tests before submitting your changes.
5. **Commit changes**:
   Write clear commit messages:
   ```bash
   git add .
   git commit -m "Fix: Resolve memory mismatch in DynamicMemoryCell"
   ```
6. **Push changes**:
   ```bash
   git push origin fix-issue-123
   ```
7. **Submit a Pull Request**:
   - Open a pull request (PR) from your forked repository to the main branch.
   - Reference the issue number (if applicable).
   - Provide a short summary and detailed description of your changes.

---

## ✨ **Coding Guidelines**

To maintain clean and consistent code:

1. **Code Style**:
   - Follow [PEP 8](https://peps.python.org/pep-0008/).
   - Use `black` for formatting:
     ```bash
     pip install black
     black .
     ```

2. **Naming Conventions**:
   - Use descriptive variable and function names.
   - Classes: `PascalCase`, Functions: `snake_case`.

3. **Type Annotations**:
   Add type hints where applicable:
   ```python
   def add_knowledge(source: str, relation: str, target: str) -> None:
       ...
   ```

4. **Comments and Docstrings**:
   - Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
   - Example:
     ```python
     def add_knowledge(source: str, relation: str, target: str) -> None:
         """
         Add a relationship between two entities to the graph.

         Args:
             source (str): The starting node.
             relation (str): The relationship type.
             target (str): The target node.
         """
         pass
     ```

---

## 🧪 **Testing and Documentation**

### **Writing Tests**:
- Add tests to the `tests/` directory.
- Use **pytest** for writing and running tests:
  ```bash
  pip install pytest
  pytest tests/
  ```
- Ensure your changes don’t break existing functionality.

### **Documentation**:
- Update relevant parts of the documentation in `README.md` or `docs/`.
- Add examples to showcase your changes.

---

## 💬 **Community**

- **Join the Conversation**: Ask questions, share ideas, or get help on our [Discord Community](https://discord.gg/zK94WvRjZT).
- **File Issues**: Share feedback or suggestions on [GitHub Issues](https://github.com/yourusername/graphfusionai/issues).
- **Stay Updated**: Follow releases and updates on GitHub.

---

## ❤️ **Thank You!**

Your contributions make **GraphFusionAI** better. Whether you're reporting bugs, writing code, or improving documentation, we appreciate your support and creativity. Together, we can build something amazing!

---

Happy Coding! 🚀
