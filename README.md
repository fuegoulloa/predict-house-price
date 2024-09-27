# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Ames Price Prediction Model

### Problem Statement

In the United States people value spacious properties. Therefore, it is assumed that the more living space is available in a house, and the bigger the lot, the higher the price the property would command in the local market. This analysis aims to build a predictive pricing model based on that premise.
The model will utilize multiple linear regression techniques, and the model's performance will be assessed using the metric Root Mean Squared Error (RMSE).
Once the model is built and refined, it will be used to predict sales prices as part of a Kaggle challenge.


-----

### Datasets

Dataset: AmesHousing.txt (original), also referred to as ames_clean.csv (cleaned up version).
All features below are part of this dataset's clean version.

|Feature|Type|Description|
|---|---|---|
|**ID**|_object_|Observation number.  String format, no units.|
|**PID**|_object_|Parcel identification number.  String format, no units.|
|**MS SubClass**|_int_|Type of dwelling.  Integer format, no units.|
|**MS Zoning**|_object_|General zoning classification.  String format, no units.|
|**Lot Frontage**|_float_|Linear feet of street connected to property.  Float format, units in feet.|
|**Lot Area**|_int_|Lot size.  Integer format, units in square feet.|
|**Street**|_object_|Type of road access to property.  String format, no units.|
|**Alley**|_object_|Type of alley access to property.  String format, no units.|
|**Lot Shape**|_object_|General shape of property.  String format, no units.|
|**Land Contour**|_object_|Flatness of the property.  String format, no units.|
|**Utilities**|_object_|Type of utilities available.  String format, no units.|
|**Lot Config**|_object_|Lot configuration.  String format, no units.|
|**Land Slope**|_object_|Slope of property.  String format, no units.|
|**Neighborhood**|_object_|Physical location within Ames city limits.  String format, no units.|
|**Condition 1**|_object_|Proximity to various conditions.  String format, no units.|
|**Condition 2**|_object_|Proximity to various conditions (if more than one is present).  String format, no units.|
|**Bldg Type**|_object_|Type of dwelling.  String format, no units.|
|**House Style**|_object_|Style of dwelling.  String format, no units.|
|**Overall Qual**|_int_|Rating of the overall material and finish of the house.  Integer format, no units.|
|**Overall Cond**|_int_|Rating of the overall condition of the house.  Integer format, no units.|
|**Year Built**|_int_|Original construction date.  Integer format, no units.|
|**Year Remod/Add**|_int_|Remodel date, same as above if NA.  Integer format, no units.|
|**Roof Style**|_object_|Type of roof.  String format, no units.|
|**Roof Matl**|_object_|Type of roof material.  String format, no units.|
|**Exterior 1**|_object_|Exterior covering on the house.  String format, no units.|
|**Exterior 2**|_object_|Exterior covering on the house if more than one type.  String format, no units.|
|**Mas Vnr Type**|_object_|Masonry veneer type.  String format, no units.|
|**Mas Vnr Area**|_float_|Masonry veneer area.  Float format, units in square feet.|
|**Exter Qual**|_object_|Rating of the material quality on the exterior.  String format, no units.|
|**Exter Cond**|_object_|Rating of present condition of exterior material.  String format, no units.|
|**Foundation**|_object_|Type of foundation.  String format, no units.|
|**Bsmt Qual**|_object_|Rating on the height of the basement.  String format, no units.|
|**Bsmt Cond**|_object_|Rating on the general condition of the basement.  String format, no units.|
|**Bsmt Exposure**|_object_|Refers to walkout or garden level walls.  String format, no units.|
|**BsmtFin Type 1**|_object_|Rating on basement finished area.  String format, no units.|
|**BsmtFin SF 1**|_float_|Type 1 finished area.  Float format, units in square feet.|
|**BsmtFinType 2**|_object_|Rating on basement finished area if more than one type.  String format, no units.|
|**BsmtFin SF 2**|_float_|Type 2 finished area.  Float format, units in square feet.|
|**Bsmt Unf SF**|_float_|Area of unfinished basement sections.  Float format, units in square feet.|
|**Total Bsmt SF**|_float_|Total area of basement.  Float format, units in square feet.|
|**Heating**|_object_|Type of heating.  String format, no units.|
|**HeatingQC**|_object_|Heating quality and condition.  String format, no units.|
|**Central Air**|_object_|Central air conditioning.  String format, no units.|
|**Electrical**|_object_|Electrical system.  String format, no units.|
|**1st Flr SF**|_int_|First floor area.  Integer format, units in square feet.|
|**2nd Flr SF**|_int_|Second floor area.  Integer format, units in square feet.|
|**Low Qual Fin SF**|_int_|Area of low quality finishes.  Integer format, units in square feet.|
|**Gr Liv Area**|_int_|Above ground living area.  Integer format, units in square feet.|
|**Bsmt Full Bath**|_int_|Number of full bathrooms in basement.  Integer format, no units.|
|**Bsmt Half Bath**|_int_|Number of half bathrooms in basement.  Integer format, no units.|
|**Full Bath**|_int_|Number of full bathroms above ground.  Integer format, no units.|
|**Half Bath**|_int_|Number of half bathroms above ground.  Integer format, no units.|
|**Bedroom**|_int_|Number of bedrooms above ground.  Integer format, no units.|
|**Kitchen**|_int_|Number of kitchens above ground.  Integer format, no units.|
|**KitchenQual**|_object_|Kitchen quality.  String format, no units.|
|**TotRmsAbvGrd**|_int_|All rooms above ground not including bathrooms.  Integer format, no units.|
|**Functional**|_object_|Home functionality.  String format, no units.|
|**Fireplaces**|_int_|Number of fireplaces.  Integer format, no units.|
|**FireplaceQu**|_object_|Fireplace quality.  String format, no units.|
|**Garage Type**|_object_|Garage location.  String format, no units.|
|**Garage Yr Blt**|_int_|Year garage was built.  Integer format, no units.|
|**Garage Finish**|_object_|Interio finish of garage.  String format, no units.|
|**Garage Cars**|_int_|Number of cars that fit in garage.  Integer format, no units.|
|**Garage Area**|_float_|Size of garage.  Float format, units in square feet.|
|**Garage Qual**|_object_|Garage quality rating.  String format, no units.|
|**Garage Cond**|_object_|Garage condition rating.  String format, no units.|
|**Paved Drive**|_object_|Paved status of driveway.  String format, no units.|
|**Wood Deck SF**|_float_|Wood deck size.  Float format, units in square feet.|
|**Open Porch SF**|_float_|Open porch area size.  Float format, units in square feet.|
|**Enclosed Porch**|_float_|Enclosed porch area size.  Float format, units in square feet.|
|**3-Ssn Porch**|_float_|Three-season porch area size.  Float format, units in square feet.|
|**Screen Porch**|_float_|Screened porch area size.  Float format, units in square feet.|
|**Pool Area**|_float_|Pool area size.  Float format, units in square feet.|
|**Pool QC**|_object_|Pool quality rating.  String format, no units.|
|**Fence**|_object_|Fence quality rating.  String format, no units.|
|**Misc Feature**|_object_|Misscelaneous features not covered in other categories.  String format, no units.|
|**Misc Val**|_float_|Dollar value of miscellaneous features.  Float format, units in USD.|
|**Mo Sold**|_int_|Month house sold.  Integer format, no units.|
|**Yr Sold**|_int_|Year house sold.  Integer format, no units.|
|**Sale Type**|_object_|Type of sale.  String format, no units.|
|**SalePrice**|_float_|Sale price.  String format, units in USD.|

---

### Summary

Based on the previous EDA exercise, it is apparent that the available data supports the initial assumption.  That is, if a developing/poor country gets foreign aid, then it has the potential to use that assistance to spur overall development and economic growth.  This growth eventually leads to increased wealth shared across the population, which has a positive impact on life expectancy.<br><br>
Although the data shows examples in support of the hypothesis, it is not universal.  Some aid-receiving countries have been unable to replicate the success of its peers. Without further analysis, one cannot draw definite conclusions.<br><br>
One hypothesis that can be explored in future iterations of this exercise is looking into other variables, such as implemented policies, governmental corruption, incidence of conflict, etc. in order to determine the key differentiator between aid-receiving countries that succeed vs. those that cannot escape poverty.  That, obviously, is beyond the scope of this analysis.

---

### Recommendation

Organizations that administer aid funds should use successful countries such as Chile or Singapore as benchmarks, and then based on each country's demographic profile and economic profile, adapt those benchmarks to local conditions. At the same time, these organizations should also study the reasons why some countries cannot kick the aid habit, such as Afghanistan or Uganda, so as to not replicate those failures.

---

### Technical Report

Complete and thorough analysis in a Jupyter notebook is located [here](./code/starter-code-ulloa.ipynb).

---


### Presentation

Accompanying presentation in PDF format is located [here](./presentation/project_01.pdf).

