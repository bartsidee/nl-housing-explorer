# Contributing

Guidelines for contributing to NL Housing Explorer.

## 🤝 How to Contribute

### Reporting Issues
- Use GitHub Issues
- Include steps to reproduce
- Include environment (OS, Python version)
- Include error messages

### Feature Requests
- Open issue with "enhancement" label
- Describe use case
- Explain benefits

### Pull Requests
1. Fork repository
2. Create branch (`git checkout -b feature/name`)
3. Make changes
4. Test thoroughly
5. Commit with clear messages
6. Open PR

---

## 📋 Development Setup

```bash
git clone https://github.com/YOUR-FORK-USERNAME/nl-housing-explorer.git
cd nl-housing-explorer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Testing

Before PR:
- [ ] All tabs load
- [ ] Maps display
- [ ] Custom scores work
- [ ] No Python errors
- [ ] Code follows style

---

## 📝 Code Style

- Follow PEP 8
- Descriptive variable names
- Add docstrings
- Comment complex logic

---

## 🎯 Areas for Contribution

**Easy:**
- New presets
- Documentation
- Typo fixes

**Medium:**
- New indicators
- Visualizations
- Performance improvements

**Advanced:**
- New data sources
- Advanced filters
- Export functionality

---

## 📊 Adding New Indicators

1. Create loader in `src/`
2. Integrate in `scripts/process_multiyear_trends.py`
3. Add to UI in `app.py`
4. Add color mapping in `components/map_viewer.py`
5. Update docs

Example:
```python
# src/your_loader.py
def load_your_data(year):
    return df

# scripts/process_multiyear_trends.py
from your_loader import load_your_data
df = df.merge(load_your_data(year), on='gwb_code_10')

# app.py
MAP_INDICATOR_BASE_OPTIONS = [
    ('Your Indicator', 'your_column'),
]
```

---

## 📜 License

Contributions licensed under MIT License.

---

## 💬 Questions?

Open GitHub issue.

---

**Thank you for contributing!**
