# Land Cover Classification and Change Detection System

## Overview

The Land Cover Classification and Change Detection System is an interactive geospatial analytics platform developed using Python and Streamlit. The application utilizes satellite imagery from Landsat Collection 2 and Sentinel-2 datasets to classify land cover, analyze spatial patterns, detect environmental changes, and visualize land-use dynamics over time.

The system enables users to explore historical and current land-cover conditions for any selected location, compare multiple years, monitor trends, and identify significant changes in vegetation, water bodies, built-up regions, and barren land.


## Objectives

* Classify land-cover categories using satellite imagery and machine learning techniques.
* Monitor land-use and land-cover changes across different years.
* Analyze long-term environmental and urbanization trends.
* Provide interactive visualizations and downloadable outputs for further analysis.
* Support decision-making in urban planning, environmental monitoring, and resource management.

## Features

### Single-Year Analysis

* Land-cover classification for a selected year and location.
* Visualization of classified maps.
* Area statistics for each land-cover category.

### Multi-Year Comparison

* Compare land-cover distributions across multiple years.
* Identify temporal variations in land-use patterns.
* Generate comparative charts and summaries.

### Trend Analysis

* Analyze long-term changes in land-cover classes.
* Track growth or decline of vegetation, water, and urban areas.
* Visualize trends through interactive plots.

### Change Detection

* Detect changes between two selected years.
* Quantify transitions between land-cover classes.
* Highlight areas experiencing significant transformation.

### Interactive Maps

* Explore classified outputs on interactive geospatial maps.
* Zoom, pan, and inspect spatial patterns dynamically.

### Data Export

* Download classification outputs, statistics, and analysis results for reporting and further processing.


## Data Sources

The project uses publicly available satellite imagery obtained through Microsoft Planetary Computer.

### Sentinel-2 Level 2A

* High-resolution multispectral imagery.
* Used for recent years and detailed land-cover analysis.

### Landsat Collection 2

* Historical satellite imagery archive.
* Used for long-term temporal analysis and change detection.



## Methodology

### Data Acquisition

Satellite imagery is retrieved from Microsoft Planetary Computer based on user-selected location and time period.

### Feature Extraction

Spectral indices and remote sensing features are generated from satellite bands, including:

* NDVI (Normalized Difference Vegetation Index)
* NDWI (Normalized Difference Water Index)
* NDBI (Normalized Difference Built-up Index)

### Land Cover Classification

The project employs machine learning techniques for land-cover classification using:

* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

These models classify pixels into major land-cover categories such as:

* Vegetation
* Water Bodies
* Built-up Areas
* Bare Land

### Change Detection

Classified maps from different years are compared to identify spatial and quantitative changes in land-cover distribution.

## Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM

### Geospatial Processing

* Rasterio
* GeoPandas
* NumPy
* Shapely

### Data Visualization

* Plotly
* Matplotlib
* Folium

### Web Application

* Streamlit

### Cloud Data Access

* Microsoft Planetary Computer

## Applications

* Environmental Monitoring
* Urban Growth Analysis
* Land Use Planning
* Resource Management
* Agricultural Assessment
* Climate and Sustainability Studies

## Future Enhancements

* Deep Learning-based land-cover classification.
* Near real-time monitoring capabilities.
* Additional land-cover categories.
* Accuracy assessment using ground-truth datasets.
* Integration of advanced geospatial analytics and forecasting models.

## Conclusion

This project demonstrates the integration of Remote Sensing, Geographic Information Systems (GIS), and Machine Learning to analyze land-cover patterns and detect environmental changes. By combining satellite imagery with interactive visual analytics, the system provides a practical solution for monitoring and understanding landscape transformations over time.
