# GitHub Configuration

## Repository Information

- **Owner**: sidnei-almeida
- **Repository**: brazilian-license-plate-recognition
- **Branch**: main

## Image URLs

The application automatically loads test images from the GitHub repository:

```
https://raw.githubusercontent.com/sidnei-almeida/brazilian-license-plate-recognition/main/images/
```

## How to Update

### Change Repository Owner/Name

Edit `app.py` lines 28-31:

```python
GITHUB_USER = "sidnei-almeida"
GITHUB_REPO = "brazilian-license-plate-recognition"
GITHUB_BRANCH = "main"
```

### Add New Test Images

1. Add images to the `images/` directory
2. Update `EXAMPLE_IMAGES` list in `app.py` (lines 35-46)
3. Commit and push to GitHub

## Benefits

✅ **No local storage needed** - Images load from GitHub  
✅ **Streamlit Cloud compatible** - Works perfectly on free tier  
✅ **Automatic caching** - Fast loading with `@st.cache_data`  
✅ **Easy updates** - Just commit new images to GitHub  

