import pandas as pd
import dask.dataframe as dd

dim_categories = 'dim_categories.csv'
category_to_product = 'category_to_product.csv'
product_to_listing = 'product_to_listing.csv'
listings_daily = 'listing_daily_p2.csv'

# find all category ids of the Toys category.
category_chunksize = 10000
category_col_names = ['Category_ID', 'Category_Path']
category_cols_numbers = [2,4]
# dataframe of category IDs of all toys categories
relevant_category_IDs = pd.DataFrame()
# relevant_category_IDs = []
General_Category = "/Toys"

for df in pd.read_csv(dim_categories,names = category_col_names,usecols = category_cols_numbers,
                      chunksize = category_chunksize, iterator=True ):
    # delete rows with at least one NA
    df = df.dropna(axis = 0, how = 'any')
    # Changes Path column to string values
    df['Category_Path'] = df['Category_Path'].astype(str)
    # DEFINE - Select only rows with specific string (Specific category)
    df = df[df['Category_Path'].str.contains(General_Category)]
    frames = [df, relevant_category_IDs]
    relevant_category_IDs = pd.concat(frames)

relevant_category_IDs.drop_duplicates(subset='Category_ID')
cat_id_list = relevant_category_IDs['Category_ID'].tolist()


###################################################################################################################################
# Find all toys products IDs
toys_prod_ids = dd.read_csv(category_to_product, names=['Category_ID','Product_ID'], usecols=[3,5])
toys_prod_ids = toys_prod_ids.loc[toys_prod_ids['Category_ID'].isin(cat_id_list)]


################################################################################################################################
# Find all toys listings IDs
prod_to_list = dd.read_csv(product_to_listing,names=['Listing_ID', 'Product_ID']
                           , usecols=[0, 3])
# join between prod-category and prod-listing
toys_prod_to_list = toys_prod_ids.set_index('Product_ID')\
    .join(prod_to_list.set_index('Product_ID'), how='inner')
toys_prod_to_list = toys_prod_to_list.reset_index()
# turn the dask dataframe into pandas dataframe
toys_prod_to_list = toys_prod_to_list.compute()


#################################################################################################################################

# Find daily details of toys listings
listing_col_names = ['Date', 'Listing_ID', 'Ordered_Quantity',
                     'Avg_Num_Of_Listings', 'Top_Rank_Share']
listing_col_numbers = [0, 1, 5, 10, 12]
list_det_df = dd.read_csv(listings_daily, names=listing_col_names,
                          usecols=listing_col_numbers, dtype={'Ordered_Quantity': float})
list_det_df = list_det_df.dropna(how='any')
list_det_df = list_det_df[list_det_df.Top_Rank_Share != 0]
# # Union
list_det_df = list_det_df.repartition(npartitions=list_det_df.npartitions // 5)
list_toys_daily = list_det_df.merge(toys_prod_to_list, on='Listing_ID', how='inner')
list_toys_daily = list_toys_daily.drop_duplicates()


# This part of code used to calculate the 100 top frequents of toys listing IDs and send them to csv format
list_top_freq = list_toys_daily.groupby(['Listing_ID'])['Date'].count().nlargest(100)
list_top_freq = list_top_freq.compute().to_frame()
list_top_freq.to_csv('Top_Frequent_Toys_Listings.csv')


top_freq_list_toys = pd.read_csv('Top_Frequent_Toys_Listings.csv',usecols=[0], names=['Listing_ID'])
top_freq_list_toys_list = top_freq_list_toys['Listing_ID'].tolist()

# Take only 100 top freq listings daily details
list_toys_top_freq_daily = list_toys_daily.loc[list_toys_daily['Listing_ID']
    .isin(top_freq_list_toys_list)]


list_toys_top_freq_daily['Demand'] = list_toys_top_freq_daily['Ordered_Quantity'] \
                                     / list_toys_top_freq_daily['Top_Rank_Share']


list_toys_top_freq_daily.to_csv('list_daily_toys_p2-*.csv')
# list_toys_top_freq_daily.to_parquet('list_daily_toys.parquet')
