from stSurvTrans import stSurvTrans

if __name__ == "__main__":
    st_path = 'path/to/spatial_transcriptomics_data/'
    bulk_path = 'path/to/bulk_data/'

    model = stSurvTrans(
        st_path=st_path, 
        bulk_path=bulk_path
    )
    model.trainVAE()
    model.trainWeibull()

    model.save()