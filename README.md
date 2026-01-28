# 🏘️ NL Housing Explorer

Interactive dashboard for exploring Dutch neighborhoods, districts, and municipalities based on 31+ indicators from official open data sources.

**Live Demo:** [https://nl-housing-explorer.streamlit.app](https://nl-housing-explorer.streamlit.app)

## ✨ Features

### 📊 Data Analysis
- **31 indicators** across 6 data sources (CBS, Politie, RIVM, PDOK)
- **18,000+ locations**: All Dutch neighborhoods (buurten), districts (wijken), and municipalities (gemeenten)
- **Multi-year trends**: Compare data from 2020-2025
- **Real-time calculations**: Custom weighted scores

### 🗺️ Interactive Visualization
- **Choropleth maps** with dynamic coloring
- **Province and municipality filters**
- **Multi-level navigation**: Country → Province → Municipality → District → Neighborhood
- **Elevation data**: NAP height visualization

### 🎯 Custom Scoring
- **Personalized weights**: Configure your own priorities
- **6 preset profiles**: Family-friendly, Urban, Affordable, etc.
- **Auto-save**: Preferences stored in browser localStorage
- **Share configurations**: URL-based sharing

### 📈 Trend Analysis
- **5-year trends**: Growth/decline indicators
- **Composite trend scores**: Weighted multi-indicator trends
- **Visual trends**: Up/down/stable indicators

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/bartsidee/nl-housing-explorer.git
cd nl-housing-explorer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (~410 MB, see DATA_DOWNLOAD_GUIDE.md for details)
python scripts/download_veiligheid_data.py  # Automated crime data
# + Manual download CBS data (see guide)

# 4. Process data
python scripts/process_multiyear_trends.py

# Optional: Force regenerate all files
# python scripts/process_multiyear_trends.py --force

# 5. Run dashboard
streamlit run app.py
```

**Documentation:**
- **Setup:** [SETUP.md](SETUP.md) - Installation and configuration
- **Data Download:** [DATA_DOWNLOAD_GUIDE.md](DATA_DOWNLOAD_GUIDE.md) - Detailed download instructions
- **Git Strategy:** [DATA_GIT_STRATEGY.md](DATA_GIT_STRATEGY.md) - What's in git and why

## 📊 Data Sources

All data comes from official Dutch open data sources:

- **CBS** (Centraal Bureau voor de Statistiek): Demographics, income, housing
- **Politie Nederland**: Crime statistics
- **PDOK**: Geographic boundaries, elevation data
- **RIVM**: Green space percentages

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete attribution and licenses.

## 🎨 Indicators

### Demographics & Housing
- Population, households, household size
- Home ownership percentage
- Children and families percentage
- Housing density

### Socio-Economic Status (SES)
- Overall SES score (percentile)
- Wealth, income, education sub-scores
- Average income
- Labor participation

### Safety
- Total crime rate (per 1,000 inhabitants)
- Burglary rate

### Proximity to Services
- Distance to schools, daycare, doctors, supermarkets
- Distance to train stations, highways, libraries

### Environment
- Green space percentage
- Water surface percentage
- Average elevation (NAP)
- Space per person

### Diversity
- Dutch, European, non-European origin percentages

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - Interactive web app framework
- **Maps**: [Folium](https://python-visualization.github.io/folium/) - Interactive choropleth maps
- **Data**: [Pandas](https://pandas.pydata.org/), [GeoPandas](https://geopandas.org/)
- **Storage**: Browser localStorage via [streamlit-js-eval](https://github.com/aghasemi/streamlit_js_eval)
- **Deployment**: [Streamlit Cloud](https://streamlit.io/cloud)

## 📁 Project Structure

```
nl-housing-explorer/
├── app.py                      # Main dashboard application
├── components/
│   └── map_viewer.py          # Map visualization component
├── src/
│   ├── custom_score.py        # Custom scoring logic
│   ├── ahn_nap_loader.py      # Elevation data loader
│   ├── veiligheid_loader.py   # Crime data loader
│   └── ...                    # Other data loaders
├── scripts/
│   ├── process_multiyear_trends.py  # Main data processing
│   └── download_veiligheid_data.py  # Crime data downloader
├── data/
│   ├── raw/                   # Raw data (not in repo, download separately)
│   ├── processed/             # Processed data (generated)
│   ├── geo/cache/ahn/         # Elevation data (included)
│   └── presets/               # Configuration presets (included)
└── requirements.txt           # Python dependencies
```

## 🔧 Development

### Adding New Indicators
1. Add loader in `src/`
2. Integrate in `scripts/process_multiyear_trends.py`
3. Add to indicator list in `app.py`
4. Add color mapping in `components/map_viewer.py`

### Custom Profiles
Create new preset in `data/presets/`:
```json
{
  "_preset": "My Profile",
  "_description": "Description",
  "custom_weights": {
    "ses_overall": 3.0,
    "crime_rate": -2.0,
    "groen_percentage": 4.0
  }
}
```

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Attribution

Data sources:
- CBS (Centraal Bureau voor de Statistiek)
- Politie Nederland
- PDOK (Publieke Dienstverlening Op de Kaart)
- RIVM (Rijksinstituut voor Volksgezondheid en Milieu)

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete list and licenses.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📧 Contact

Questions or suggestions? Open an issue on GitHub.

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Built with ❤️ using open data from the Netherlands**
