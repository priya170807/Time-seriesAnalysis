# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:26:17 2020

@author: shvpr
"""

from io import StringIO
from datetime import date
from datetime import datetime
import time
import pandas as pd
from util import notebook_utils
from util import fcst_utils
import dateutil.parser
today=date.today()
status_indicator = notebook_utils.StatusIndicator()

# AWS libraries
import boto3


def main():
    

    # AWS credentials through named profile.
    boto3.setup_default_session(profile_name='HCL_USER_1')
    
    # Bucket to store the cleaned data for amazon forecast
    bucket_name="test-forecast-example"
    prefix="capplan_data/"
    key=prefix+"capplan-time-train"+str(today)+".csv"
    role_name="ForecastAccessS3Role"
    role_arn="arn:aws:iam::311461407286:role/ForecastAccessS3Role"
    project = 'capacity_planning_forecast_deepAR'
    predictorName= project+'_deepAR_algo'
    algorithmArn = 'arn:aws:forecast:::algorithm/Deep_AR_Plus'
    forecastName= project+'_deepAR_algo_forecast'
    forecastHorizon = 288
    
    # Bucket containing the input data
    # mybucket = "capacityplanning"
    # myprefix="cloudwatch_metricdata/"
    
    # Bucket to store the predictions in the form of csv files
    new_bucket="capacityplanning-predictions-bucket"
    s3_output_path = "predictions/"
    Deep_Bytes_key = s3_output_path+"DEEPAR_WRITEBYTES"+str(today)+".csv"
    Deep_Ops_key = s3_output_path+"DEEPAR_WRITEOPS"+str(today)+".csv"
    RMSE_key = s3_output_path+"RMSE_DeepAR"+str(today)+".txt"
    
    #DeepAR_key = s3_output_path+"PREDICTIONS_FUTURE_DEEPAR"+str(today)+".csv"
    
    # Setting up clients for S3, forecast, forecastquery.
    client = boto3.client('s3')
    forecast=boto3.client('forecast')
    forecastquery=boto3.client('forecastquery')
    
    # copy to output bucket function
    def copy_to_s3(client, df, bucket, filepath):
            csv_buf = StringIO()
            df.to_csv(csv_buf, header=True, index=False)
            csv_buf.seek(0)
            client.put_object(Bucket=bucket, Body=csv_buf.getvalue(), Key=filepath)
            print(f'Copy {df.shape[0]} rows to S3 Bucket {bucket} at {filepath}, Done!')
    # Preparing data for the forecast
    def initialize_data_forecast():
        DATASET_FREQUENCY = "5min" 
        TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"
        datasetName= project+'_ds'
        datasetGroupName= project +'_dsg'
        s3DataPath = "s3://"+bucket_name+"/"+key
        create_dataset_group_response = forecast.create_dataset_group(DatasetGroupName=datasetGroupName,
                                                                  Domain="CUSTOM",
                                                                 )
        datasetGroupArn = create_dataset_group_response['DatasetGroupArn']
        schema ={
       "Attributes":[
          {
             "AttributeName":"timestamp",
             "AttributeType":"timestamp"
          },
          {
             "AttributeName":"item_id",
             "AttributeType":"string"
          },
          {
             "AttributeName":"target_value",
             "AttributeType":"float"
          }
       ]
    }
        response=forecast.create_dataset(
                        Domain="CUSTOM",
                        DatasetType='TARGET_TIME_SERIES',
                        DatasetName=datasetName,
                        DataFrequency=DATASET_FREQUENCY, 
                        Schema = schema)
        datasetArn=response['DatasetArn']
        forecast.update_dataset_group(DatasetGroupArn=datasetGroupArn, DatasetArns=[datasetArn])
    
        return datasetGroupArn, datasetArn, datasetName, project, s3DataPath, DATASET_FREQUENCY, TIMESTAMP_FORMAT
    
    # Creating data import jobs
    
    # role_name="ForecastAccessS3Role"
    # role_arn="arn:aws:iam::311461407286:role/ForecastAccessS3Role"
    def create_data_imports(forecastHorizon, predictorName, algorithmArn, datasetGroupArn, datasetArn):
        datasetImportJobName = 'EP_DSIMPORT_JOB_TARGET'
        ds_import_job_response=forecast.create_dataset_import_job(DatasetImportJobName=datasetImportJobName,
                                                              DatasetArn=datasetArn,
                                                              DataSource= {
                                                                  "S3Config" : {
                                                                     "Path":s3DataPath,
                                                                     "RoleArn": role_arn
                                                                  } 
                                                              },
                                                              TimestampFormat=TIMESTAMP_FORMAT
                                                             )
        ds_import_job_arn=ds_import_job_response['DatasetImportJobArn']
        status_indicator=notebook_utils.StatusIndicator()
        while True:
            status1 = forecast.describe_dataset_import_job(DatasetImportJobArn=ds_import_job_arn)['Status']
            status_indicator.update(status1)
            if status1 in ('ACTIVE', 'CREATE_FAILED'): break
            time.sleep(15)
        status_indicator.end()
        assert status1 == 'ACTIVE', ("The status is not active, it is failed")
        return status1, ds_import_job_arn
            
    
    def create_predictor(predictorName, algorithmArn, forecastHorizon, datasetGroupArn):
        create_predictor_response=forecast.create_predictor(PredictorName=predictorName, 
                                                      AlgorithmArn=algorithmArn,
                                                      ForecastHorizon=forecastHorizon,
                                                      PerformAutoML= False,
                                                      PerformHPO=True,
                                                      EvaluationParameters= {"NumberOfBacktestWindows": 1, 
                                                                             "BackTestWindowOffset": 288}, 
                                                      HPOConfig={
                                                          'ParameterRanges':{
                                                          'ContinuousParameterRanges':[
                                                                {
                                                                  'Name':'learning_rate',
                                                                  'MaxValue':0.1,
                                                                  'MinValue':0.01,
                                                                  'ScalingType':'Linear'
                                                                  }
                                                               ],
                                                          'IntegerParameterRanges':[
                                                              {
                                                                  'Name':'context_length',
                                                                  'MaxValue':100,
                                                                  'MinValue':50,
                                                                  'ScalingType':'Linear'
                                                              }
                                                              ]
                                                             }
                                                        },
                                                      InputDataConfig= {"DatasetGroupArn": datasetGroupArn},
                                                      FeaturizationConfig= {"ForecastFrequency": "5min", 
                                                                            "Featurizations": 
                                                                            [
                                                                              {"AttributeName": "target_value", 
                                                                               "FeaturizationPipeline": 
                                                                                [
                                                                                  {"FeaturizationMethodName": "filling", 
                                                                                   "FeaturizationMethodParameters": 
                                                                                    {"frontfill": "none", 
                                                                                     "middlefill": "zero", 
                                                                                     "backfill": "zero"}
                                                                                  }
                                                                                ]
                                                                              }
                                                                            ]
                                                                           }
                                                     )
        predictor_arn=create_predictor_response['PredictorArn']
        #forecast.get_accuracy_metrics(PredictorArn=predictor_arn)
        status_indicator=notebook_utils.StatusIndicator()
        while True:
            status2 = forecast.describe_predictor(PredictorArn=predictor_arn)['Status']
            status_indicator.update(status2)
            if status2 in ('ACTIVE', 'CREATE_FAILED'): break
            time.sleep(15)
        status_indicator.end()
        assert status2=='ACTIVE', ("The status is not active,it is failed")
        return status2, predictor_arn
    
    def create_forecast(forecastName, predictor_arn):
        create_forecast_response=forecast.create_forecast(ForecastName=forecastName,
                                                          PredictorArn=predictor_arn)
        forecast_arn = create_forecast_response['ForecastArn']
        status_indicator=notebook_utils.StatusIndicator()
        while True:
            status3 = forecast.describe_forecast(ForecastArn=forecast_arn)['Status']
            status_indicator.update(status3)
            if status3 in ('ACTIVE', 'CREATE_FAILED'): break
            time.sleep(15)
        status_indicator.end()
        assert status3=='ACTIVE', ("The status is not active,it is failed")
        return status3, forecast_arn
            
    
    def create_final_predictions(status3, forecast_arn):
        
    
        forecastResponse_Bytes = forecastquery.query_forecast(ForecastArn=forecast_arn,
                                                                Filters={"item_id":"EBSWriteBytes"}
                                                                )
        forecastResponse_Ops = forecastquery.query_forecast(ForecastArn=forecast_arn,
                                                                Filters={"item_id":"EBSWriteOps"}
                                                                )
                
        prediction_df_Bytes = pd.DataFrame.from_dict(forecastResponse_Bytes['Forecast']['Predictions']['p50'])
                
        prediction_df_Ops = pd.DataFrame.from_dict(forecastResponse_Ops['Forecast']['Predictions']['p50'])
        
        return prediction_df_Bytes, prediction_df_Ops 
                
            
    def create_result_dataframe(prediction_df_Bytes, prediction_df_Ops):
        results_df_Bytes = pd.DataFrame(columns=['timestamp', 'value', 'source'])
        results_df_Ops = pd.DataFrame(columns=['timestamp', 'value', 'source'])
        for index, row in prediction_df_Bytes.iterrows():
            clean_timestamp = dateutil.parser.parse(row['Timestamp'])
            results_df_Bytes = results_df_Bytes.append({'timestamp' : clean_timestamp , 'value' : row['Value'], 'source': 'bytes_p50'} , ignore_index=True)
        for index, row in prediction_df_Ops.iterrows():
            clean_timestamp = dateutil.parser.parse(row['Timestamp'])
            results_df_Ops = results_df_Ops.append({'timestamp' : clean_timestamp , 'value' : row['Value'], 'source': 'Ops_p50'} , ignore_index=True)
        #pivot_df_Ops = results_df_Ops.pivot(columns='source', values='value', index="timestamp")
        #pivot_df_Bytes = results_df_Bytes.pivot(columns='source', values='value', index="timestamp")
        return results_df_Bytes, results_df_Ops
        
    def clean_up_resources(forecast_arn, predictor_arn, ds_import_job_arn, datasetArn, datasetGroupArn):
        fcst_utils.wait_till_delete(lambda: forecast.delete_forecast(ForecastArn=forecast_arn))
        #print("forecast job and forecastArn cleaned up....")
        fcst_utils.wait_till_delete(lambda: forecast.delete_predictor(PredictorArn=predictor_arn))
        #print("Predictor job and PredictorArn cleaned up....")
        fcst_utils.wait_till_delete(lambda: forecast.delete_dataset_import_job(DatasetImportJobArn=ds_import_job_arn))
        #print("dataset import job cleaned up....")
        fcst_utils.wait_till_delete(lambda: forecast.delete_dataset(DatasetArn=datasetArn))
        #print("create dataset job and datasetArn cleaned up....")
        fcst_utils.wait_till_delete(lambda: forecast.delete_dataset_group(DatasetGroupArn=datasetGroupArn))
        #print("datasetgroup job and datasetgroupArn cleaned up....")



    #df = compound_dataframe(mybucket)
    #preprocess_data(df)
    datasetGroupArn, datasetArn, datasetName, project, s3DataPath, DATASET_FREQUENCY, TIMESTAMP_FORMAT=initialize_data_forecast()
    
    try:
        status1, ds_import_job_arn=create_data_imports(forecastHorizon, predictorName, algorithmArn, datasetGroupArn, datasetArn)
        print("The value of status1: ", status1)
    except AssertionError as error:
        print(error)
    else:
        try:
            status2, predictor_arn=create_predictor(predictorName, algorithmArn, forecastHorizon, datasetGroupArn)
            print("The value of status2: ", status2)
        except AssertionError as error:
            print(error)
        else:
            try:
                status3, forecast_arn=create_forecast(forecastName, predictor_arn)
                print("The value of status3: ", status3)
            except AssertionError as error:
                print(error)
            else:
                
                prediction_df_Bytes, prediction_df_Ops=create_final_predictions(status3, forecast_arn)
                results_df_Bytes, results_df_Ops=create_result_dataframe(prediction_df_Bytes, prediction_df_Ops)
                
    
    
    
    
    copy_to_s3(client=client, df=results_df_Bytes, bucket=new_bucket, filepath=Deep_Bytes_key)  
    copy_to_s3(client=client, df=results_df_Ops, bucket=new_bucket, filepath=Deep_Ops_key)
             
    res = forecast.get_accuracy_metrics(PredictorArn=predictor_arn)
    RMSE = res['PredictorEvaluationResults'][0]['TestWindows'][0]['Metrics']['RMSE']
    with open("rmse.txt", "w") as file:
        file.write(str(RMSE))
    client.upload_file(Filename='rmse.txt', Bucket= new_bucket, Key=RMSE_key)
    clean_up_resources(forecast_arn, predictor_arn, ds_import_job_arn, datasetArn, datasetGroupArn)
    endtime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("cron ended at: {}".format(endtime))

if __name__=="__main__":
    main()
    
                
