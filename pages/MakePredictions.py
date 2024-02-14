import streamlit as st
import os
import requests
import numpy as np
from tensorflow.keras.models import load_model
import joblib
###############################""
st.set_page_config(
    page_title="MakePredictions",
    page_icon="static/icon.png",  
)
################################### Define used variables and Functions ############
# Max request per day up to 100000 calls
api_key='AZ8DNA6CDB1X1XAKUFCCN4H7J7HBCFKVZQ'
# Specify the API endpoint URL
api_url = 'https://api.etherscan.io/api'
# Get the current directory path
current_dir = os.path.dirname(__file__)
# Construct the path to  model files in folder Models
LSTMmodelFilePath = os.path.join(current_dir, '..', 'Models', 'trained_lstm_model.h5') # path LSTM model
MLPFrontmodelFilePath = os.path.join(current_dir, '..', 'Models', 'trained_MLPFrontTx_model.h5') # Path MLP model for Front-runing Tx
MLPBackmodelFilePath = os.path.join(current_dir, '..', 'Models', 'modelMLPBackTx.h5') # path MLP model for Back-runing Tx
# Construct the path to scaler files in folder Scalers
scalerInputsLstmPath = os.path.join(current_dir, '..', 'Scalers', 'scaler_inputsLSTM.pkl') # path Scaler of  inputs for LSTM
scalerOutputLstmPath = os.path.join(current_dir, '..', 'Scalers', 'scaler_OutputLSTM.pkl') # path Scaler of  output for LSTM
scalerInputsMLPPath = os.path.join(current_dir, '..', 'Scalers', 'scaler_inputsMLP.pkl') # path Scaler of  inputs for MLP model
scalerOutputMLPFrontPath = os.path.join(current_dir, '..', 'Scalers', 'scaler_FrontRuningOutputMLP.pkl') # path Scaler of  output for MLP related to Front-runing Tx
scalerOutputMLPBackPath = os.path.join(current_dir, '..', 'Scalers', 'scaler_BackRuningOutputMLP.pkl') # path Scaler of  output for MLP related to Back-runing Tx
#######
def callTxConfTimeApiEndpoint(gasPriceTargetedTx):
   # Specify the parameters for the API request that will be used for getting estimated confirmation time of TargetedTx
    params_TxConfTime = {
    'module': 'gastracker',
    'action': 'gasestimate',
    'gasprice':int(gasPriceTargetedTx),# Gas price of targeted transaction in wei (the price paid per unit of gas, in wei / gasBaseFee+MaxPriority)
    'apikey': api_key,
    }
   # call api endpoint
    response = requests.get(api_url, params=params_TxConfTime)
   # Check if the request was successful (status code 200)
    if response.status_code == 200:
    # Parse the JSON response
      data = response.json()
      ConfTxTime= data['result'] # in seconds
      return ConfTxTime
    else:
      #st.error(f"There is a problem when calling API endpoints to get confirmation time of the targeted transaction: {response.status_code} - {response.text}. Don't worry, please try making the request again!")
      return None
######  
def callRecentBlockNumberApiEndpoint():
   # Specify the parameters for the API request that will be used for getting number of most recent block 
    params_RecentBlockNumber = {
    'module': 'proxy',
    'action': 'eth_blockNumber', 
    'apikey': api_key,
    }
   # call api endpoint
    response = requests.get(api_url, params=params_RecentBlockNumber)
   # Check if the request was successful (status code 200)
    if response.status_code == 200:
    # Parse the JSON response
      data = response.json()
      BlockNbr= data['result'] # Block number in Hex should be converted to decimal
      return int(BlockNbr, 16)
    else:
      #st.error(f"There is a problem when calling API endpoints to get Number of most recent Mined or Stacked Block: {response.status_code} - {response.text}. Don't worry, please try making the request again!")
      return None
#####  
def callGetBlockByNumberEndpoint(BlockNbr):
    # Specify the parameters for the API request
    paramsGetBlockByNumberEndpoint = {
    'module': 'proxy',
    'action': 'eth_getBlockByNumber',
    'tag':hex(BlockNbr),
    'boolean':'false',
    'apikey': api_key,
    }
   # call api endpoint
    response = requests.get(api_url, params=paramsGetBlockByNumberEndpoint)
   # Check if the request was successful (status code 200)
    if response.status_code == 200:
    # Parse the JSON response
      data = response.json()
      BlockTimestamp= data['result']['timestamp'] # Get TimeStamp of the block in hex format
      GasUsed=data['result']['gasUsed'] # Get GasUsed within the block in hex format
      BlockSize=data['result']['size'] # Get Size of the block in hex format
      NbrTx=len(data['result']['transactions']) # Get Number of transactions in the block
      BaseGasFee= data['result']['baseFeePerGas'] # Get BaseGasFee of the block in hex format
      return int(BlockTimestamp,16),int(GasUsed,16),int(BlockSize,16),NbrTx,int(BaseGasFee,16)/1000000000
    else:
      #st.error(f"There is a problem when calling this following API endpoint GetBlockByNumberEndpoint: {response.status_code} - {response.text}. Don't worry, please try making the request again!")
      return None
#####  
def callBlockTimeCountApiEndpoint(BlockNbr):
    # Specify the parameters for the API request that will be used for getting  estimated time remaining, in seconds, until a certain block is mined or stacked
    params_BlockTimeCount = {
    'module': 'block',
    'action': 'getblockcountdown',
    'blockno':BlockNbr,
    'apikey': api_key,
   }
   # call api endpoint
    response = requests.get(api_url, params=params_BlockTimeCount)
   # Check if the request was successful (status code 200)
    if response.status_code == 200:
    # Parse the JSON response
      data = response.json()
      BlockTimeCount= data['result']['EstimateTimeInSec'] # Estimated Time in Seconds to stacked the block 
      return BlockTimeCount
    else:
      #st.error(f"There is a problem when calling API endpoints to get estimated time remaining, in seconds, until a certain block is mined or stacked: {response.status_code} - {response.text}. Don't worry, please try making the request again!")
      return None  
######### Validate BackRuning and Front runing Tx function
def CheckingFront(txdiff,LimitGaz): 
         i = 0.65*LimitGaz ## 75% of mean (Difference MaxPriority TargetedTX FrontTx)  front runing Priority distribution of front runing tx in Gwei
         cou=0
         while txdiff >= 0:
            txdiff -= i 
            cou+=1 
         return cou*i

def CheckingBack(txdiff,LimitGaz): 
         i = 0.065*LimitGaz # 95% of percentile 75% (Difference MaxPriority TargetedTX BackRuningTx)
         cou=0
         while txdiff >= 0:
            txdiff -= i 
            cou+=1 
         return cou*i


#########################################################
# Initialize session state
if 'TargetedBlock' not in st.session_state:
    st.session_state.TargetedBlock = 0
if 'remainingblock' not in st.session_state:   
    st.session_state.remainingblock = 0
if 'EstimatedConfirmationTime' not in st.session_state:
    st.session_state.EstimatedConfirmationTime = 0
if 'ClickedBlockNumberButton' not in st.session_state:
    st.session_state.ClickedBlockNumberButton=False
if 'Gp' not in st.session_state:
    st.session_state.Gp=0
if 'PredictedBaseGasFee' not in st.session_state:
    st.session_state.PredictedBaseGasFee=0
if 'BlockNumberInput' not in st.session_state:
    st.session_state.BlockNumberInput=0
if 'ClickedBaseGasButton' not in st.session_state:
    st.session_state.ClickedBaseGasButton=False
if 'Remaining' not in st.session_state:            
    st.session_state.Remaining=0

########################## Page config   
st.title("MevStrategy Predictions 🔍🥪📈")
st.markdown(""" 
            ### Get estimated Block that in wich the Targeted Transaction will be registred or confirmed  : 
            """)
# Use st.form to isolate the button logic
with st.form("estimate_block_form"):
    st.warning("Please enter the gas price of the targeted transaction in wei, Gas price is Max Transaction fee divised by LimitGaz of transactrion Ok or the price paid per unit of gas in wei, don't forget that to avoid misleading estimation")
    # Add input field for gas price of targeted transaction
    GasPrice = st.number_input("Gas Price (wei)", step=1)
    BlockButton = st.form_submit_button("Estimate Block Number")
    if BlockButton:
      RecentBlockNumber=callRecentBlockNumberApiEndpoint()
      EstimatedConfirmationTime=float(callTxConfTimeApiEndpoint(int(GasPrice)))
      remainingblock = int(EstimatedConfirmationTime/12)
      st.session_state.EstimatedConfirmationTime = EstimatedConfirmationTime
      st.session_state.remainingblock = remainingblock
      st.session_state.TargetedBlock = RecentBlockNumber + remainingblock
      st.session_state.ClickedBlockNumberButton=True
      st.session_state.Gp=int(GasPrice)

# Input field for Gas price of targeted Transaction
# st.warning("Please enter the gas price of the targeted transaction in wei, Gas price is Max Transaction fee divised by LimitGaz of transactrion Ok, don't forget that to avoid misleading estimation")
# GasPrice = st.number_input("Gas Price (wei)",step=1)
# BlockButton= st.button("Estimate Block Number",key='EstimateBlockNumber')
# Button to estimate block number
if st.session_state.ClickedBlockNumberButton:
  st.success(f"Estimated Confirmation Time of Targeted Tx is : {st.session_state.EstimatedConfirmationTime} seconds")
  st.warning(f"Actual BlockNumber is : {st.session_state.TargetedBlock-st.session_state.remainingblock}")
  st.success(f"The estimated Block that in wich the Targeted Transaction will be registred or confirmed is : {st.session_state.TargetedBlock}")

st.markdown(""" 
            ### Prediction of BaseGasFee for Targeted Block : 
            """)
with st.form("estimate_BaseGasFee_form"):
  button_=st.form_submit_button("Click to see the prediction of BaseGasFee")
  if button_ : #### check if button_ is clicked or not
          BlockNumberInput=callRecentBlockNumberApiEndpoint()
          if st.session_state.TargetedBlock-BlockNumberInput==1:
            BlockTimestampInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput=callGetBlockByNumberEndpoint(BlockNumberInput)
            Inputs=np.array([BlockTimestampInput,BlockNumberInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput]).reshape(-1,6)
            modelLstm = load_model(LSTMmodelFilePath) # Load LSTM model
            scalerinptLstm= joblib.load(scalerInputsLstmPath) # Load the inputs scaler for lstm model
            scalerLstmOutput=joblib.load(scalerOutputLstmPath) # Load the output scaler for lstm model
            Inputs=scalerinptLstm.transform(Inputs) # Scaling Inputs by loaded MinMaxScaler
            ScaledPredictedBaseGasFee =  modelLstm.predict(Inputs.reshape(-1,1,6)).reshape(-1,1)
            PredictedBaseGasFee=scalerLstmOutput.inverse_transform(ScaledPredictedBaseGasFee)
            st.session_state.PredictedBaseGasFee=PredictedBaseGasFee.reshape(-1)[0]
            st.session_state.BlockNumberInput=BlockNumberInput
            st.session_state.Remaining=1
            st.session_state.ClickedBaseGasButton=True
          elif st.session_state.TargetedBlock-int(BlockNumberInput)<=0:
            st.error("we cannot predict gaseBaseFee for targeted block by same data or predict gaseBaseFee for targeted block based on data of next block!!!")
            st.session_state.Remaining=0
          elif st.session_state.TargetedBlock-int(BlockNumberInput)==2:
            BlockTimestampInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput=callGetBlockByNumberEndpoint(BlockNumberInput)
            Inputs=np.array([BlockTimestampInput,BlockNumberInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput]).reshape(-1,6)
            modelLstm = load_model(LSTMmodelFilePath) # Load LSTM model
            scalerinptLstm= joblib.load(scalerInputsLstmPath) # Load the inputs scaler for lstm model
            scalerLstmOutput=joblib.load(scalerOutputLstmPath) # Load the output scaler for lstm model
            Inputs=scalerinptLstm.transform(Inputs) # Scaling Inputs by loaded MinMaxScaler
            ScaledPredictedBaseGasFee =  modelLstm.predict(Inputs.reshape(-1,1,6)).reshape(-1,1)
            PredictedBaseGasFee=scalerLstmOutput.inverse_transform(ScaledPredictedBaseGasFee) 
            st.session_state.PredictedBaseGasFee=PredictedBaseGasFee.reshape(-1)[0]
            st.session_state.BlockNumberInput=BlockNumberInput
            st.session_state.Remaining=2
            st.session_state.ClickedBaseGasButton=True
          elif st.session_state.TargetedBlock-int(BlockNumberInput)==3:
            BlockTimestampInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput=callGetBlockByNumberEndpoint(BlockNumberInput)
            Inputs=np.array([BlockTimestampInput,BlockNumberInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput]).reshape(-1,6)            
            modelLstm = load_model(LSTMmodelFilePath) # Load LSTM model
            scalerinptLstm= joblib.load(scalerInputsLstmPath) # Load the inputs scaler for lstm model
            scalerLstmOutput=joblib.load(scalerOutputLstmPath) # Load the output scaler for lstm model
            Inputs=scalerinptLstm.transform(Inputs) # Scaling Inputs by loaded MinMaxScaler
            ScaledPredictedBaseGasFee =  modelLstm.predict(Inputs.reshape(-1,1,6)).reshape(-1,1)
            PredictedBaseGasFee=scalerLstmOutput.inverse_transform(ScaledPredictedBaseGasFee) 
            st.session_state.PredictedBaseGasFee=PredictedBaseGasFee.reshape(-1)[0]
            st.session_state.BlockNumberInput=BlockNumberInput
            st.session_state.Remaining=3
            st.session_state.ClickedBaseGasButton=True
          elif st.session_state.TargetedBlock-int(BlockNumberInput)==4:
            BlockTimestampInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput=callGetBlockByNumberEndpoint(BlockNumberInput)
            Inputs=np.array([BlockTimestampInput,BlockNumberInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput]).reshape(-1,6)            
            modelLstm = load_model(LSTMmodelFilePath) # Load LSTM model
            scalerinptLstm= joblib.load(scalerInputsLstmPath) # Load the inputs scaler for lstm model
            scalerLstmOutput=joblib.load(scalerOutputLstmPath) # Load the output scaler for lstm model
            Inputs=scalerinptLstm.transform(Inputs) # Scaling Inputs by loaded MinMaxScaler
            ScaledPredictedBaseGasFee =  modelLstm.predict(Inputs.reshape(-1,1,6)).reshape(-1,1)
            PredictedBaseGasFee=scalerLstmOutput.inverse_transform(ScaledPredictedBaseGasFee) 
            st.session_state.PredictedBaseGasFee=PredictedBaseGasFee.reshape(-1)[0]
            st.session_state.BlockNumberInput=BlockNumberInput
            st.session_state.Remaining=4
            st.session_state.ClickedBaseGasButton=True
          elif st.session_state.TargetedBlock-int(BlockNumberInput)==5:
            BlockTimestampInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput=callGetBlockByNumberEndpoint(BlockNumberInput)
            Inputs=np.array([BlockTimestampInput,BlockNumberInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput]).reshape(-1,6)
            modelLstm = load_model(LSTMmodelFilePath) # Load LSTM model
            scalerinptLstm= joblib.load(scalerInputsLstmPath) # Load the inputs scaler for lstm model
            scalerLstmOutput=joblib.load(scalerOutputLstmPath) # Load the output scaler for lstm model
            Inputs=scalerinptLstm.transform(Inputs) # Scaling Inputs by loaded MinMaxScaler
            ScaledPredictedBaseGasFee =  modelLstm.predict(Inputs.reshape(-1,1,6)).reshape(-1,1)
            PredictedBaseGasFee=scalerLstmOutput.inverse_transform(ScaledPredictedBaseGasFee) 
            st.session_state.PredictedBaseGasFee=PredictedBaseGasFee.reshape(-1)[0]
            st.session_state.BlockNumberInput=BlockNumberInput
            st.session_state.Remaining=5
            st.session_state.ClickedBaseGasButton=True
          else:
            BlockTimestampInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput=callGetBlockByNumberEndpoint(BlockNumberInput)
            Inputs=np.array([BlockTimestampInput,BlockNumberInput,GasUsedInput,BlockSizeInput,NbrTxInput,BaseGasFeeInput]).reshape(-1,6)            
            modelLstm = load_model(LSTMmodelFilePath) # Load LSTM model
            scalerinptLstm= joblib.load(scalerInputsLstmPath) # Load the inputs scaler for lstm model
            scalerLstmOutput=joblib.load(scalerOutputLstmPath) # Load the output scaler for lstm model
            Inputs=scalerinptLstm.transform(Inputs) # Scaling Inputs by loaded MinMaxScaler
            ScaledPredictedBaseGasFee =  modelLstm.predict(Inputs.reshape(-1,1,6)).reshape(-1,1)
            PredictedBaseGasFee=scalerLstmOutput.inverse_transform(ScaledPredictedBaseGasFee) 
            st.session_state.PredictedBaseGasFee=PredictedBaseGasFee.reshape(-1)[0]
            st.session_state.BlockNumberInput=BlockNumberInput
            st.session_state.Remaining=6
            st.session_state.ClickedBaseGasButton=True

if st.session_state.ClickedBaseGasButton: # check is button related to predictBaseGasFee is clicked
  if st.session_state.Remaining==1:
      st.success('The predicted BaseGasFee in Gwei for next block {} is {}'.format(st.session_state.BlockNumberInput+1,st.session_state.PredictedBaseGasFee))
      st.warning('Remaining Blocks is 1. You have top best estimation with confidence of 95%, the error will be less than 0.0061 Gwei. But you have mostly less than 12s to execute front runing and back runing transaction, Hurry Hurry up if you can!!!!') 
  elif st.session_state.Remaining==2:
      st.success('The predicted BaseGasFee in Gwei for next block {} is {}'.format(st.session_state.BlockNumberInput+1,st.session_state.PredictedBaseGasFee))
      st.warning('Remaining Blocks is 2. You have a good estimation with confidence of 95%, the error will be less than 1.60 Gwei, but you have mostly less than 24s to execute front runing and back runing transaction, Hurry up!!!!')
  elif st.session_state.Remaining==3:
      st.success('The predicted BaseGasFee in Gwei for next block {} is {}'.format(st.session_state.BlockNumberInput+1,st.session_state.PredictedBaseGasFee))
      st.warning('Remaining Blocks is 3. You have a good estimation with confidence of 95%, the error will be less than 1.95 Gwei. You have mostly less than 36s to execute front runing and back runing transaction, Hurry up!!!!')
  elif st.session_state.Remaining==4:
      st.success('The predicted BaseGasFee in Gwei for next block {} is {}'.format(st.session_state.BlockNumberInput+1,st.session_state.PredictedBaseGasFee))
      st.warning('Remaining Blocks is 4. You have a good estimation with confidence of 95%, the error will be less than 2.21 Gwei. You have mostly less than 48s to execute front runing and back runing transaction, Hurry up!!!!')
  elif st.session_state.Remaining==5:
      st.success('The predicted BaseGasFee in Gwei for next block {} is {}'.format(st.session_state.BlockNumberInput+1,st.session_state.PredictedBaseGasFee))
      st.warning('Remaining Blocks is 5. You have a good estimation with confidence of 95%, the error will be less than 2.42 Gwei. You have mostly less than 60s to execute front runing and back runing transaction, Hurry up !!!!')
  elif st.session_state.Remaining==6:
      st.success('The predicted BaseGasFee in Gwei for next block {} is {}'.format(st.session_state.BlockNumberInput+1,st.session_state.PredictedBaseGasFee))
      st.warning("At least Remaining Blocks is 6. The Estimation can be misleading, with a minimum maximum error of 18 Gwei. You have at least 72 seconds to execute front-running and back-running transactions, but we strongly advise against executing such transactions to avoid potential losses, Please wait at least 12s and click again on button to predict gasBaseFee with accurate result !!!")
  else:
      st.error("We cannot predict the gasBaseFee for the targeted block because the actual block in Ethereum may be the same or newer than the targeted block for which you are trying to predict its gasBaseFee.")
st.markdown(""" 
            ### Prediction of MaxPriority or Priority fee of Front-runing and Back-runing Tx: 
            """)
st.warning("Don't predict MaxPriority of Back and Front runing tx if you have error message above!!!")
# Add input fields to the form
st.warning("Please enter max priority denominated in Gwei of targeted transaction ok, max priority is max amount added per gas unit a user is willing to pay for his transaction, it's not Tx fee be aware!!! ")
with st.form("estimate_MaxPriorityFBRtx_form"):
  max_priority = st.number_input("MaxPriorityOfTargetedTransaction in Gwei",min_value=0)
  base_gas_fee = st.session_state.PredictedBaseGasFee
  limit_gas = st.number_input("LimitGas of Targeted Transaction",min_value=0)
  # Add a button to submit the form
  submitted = st.form_submit_button("Predict Priority of Front and Back Running Transactions")
  # If button_ is clicked
  if submitted:
    Inputs=np.array([float(max_priority),base_gas_fee,int(limit_gas)]).reshape(-1,3)
    modelFrontRuning = load_model(MLPFrontmodelFilePath)
    modelBackRuning=load_model(MLPBackmodelFilePath)
    scalerOutputBackRuning= joblib.load(scalerOutputMLPBackPath) 
    scalerInputs= joblib.load(scalerInputsMLPPath) 
    scalerOutputFrontRuning= joblib.load(scalerOutputMLPFrontPath) 
    Inputs=scalerInputs.transform(Inputs) # Scaling Inputs by loaded MinMaxScaler
    ScaledPredictedPriorityFront = modelFrontRuning.predict(Inputs).reshape(-1,1)
    ScaledPredictedPriorityBack = modelBackRuning.predict(Inputs).reshape(-1,1)
    PredictedFront=scalerOutputFrontRuning.inverse_transform(ScaledPredictedPriorityFront) 
    PredictedBack=scalerOutputBackRuning.inverse_transform(ScaledPredictedPriorityBack)
    FrontRuningTxMaxFee=(PredictedFront.reshape(-1)[0]+base_gas_fee)*int(limit_gas)
    BackRuningTxMaxFee=(PredictedBack.reshape(-1)[0]+base_gas_fee)*int(limit_gas)
    TxTargetedFee=((st.session_state.Gp*int(limit_gas))/1000000000) # convert from wei to Gwei
    difFront=TxTargetedFee-FrontRuningTxMaxFee
    difBack=BackRuningTxMaxFee-TxTargetedFee
    if difFront>=0: #check if front runing tx max fee less than targeted tx max fee
      AddedGasFee=CheckingFront(difFront,int(limit_gas))
      FrontRuningTxMaxFee=FrontRuningTxMaxFee+AddedGasFee
      FrontPriorityFeeperUnitGaz=PredictedFront.reshape(-1)[0]+(AddedGasFee/int(limit_gas))
      st.success(f'The transaction is most likely to be included in the same block of targeted Tx with 99% and before TargetedTx, with probability more than 85%: {st.session_state.TargetedBlock} before Targeted Transaction ✔️👌 / FrontRuningTx: LimitGaz: {limit_gas} ; GasPrice in Gwei: {FrontPriorityFeeperUnitGaz+base_gas_fee}  ;  max Total Fee in Gwei (BaseGasFee+MaxPriority)*LimitGaz: {FrontRuningTxMaxFee}')
    else:
      st.success(f'The transaction is most likely to be included in the same block of targeted Tx with 99% and before TargetedTx directly, with probability more than 85%: {st.session_state.TargetedBlock} before Targeted Transaction ✔️👌 / FrontRuningTx: LimitGaz: {limit_gas} ; GasPrice in Gwei: {PredictedFront.reshape(-1)[0]+base_gas_fee} ; max Total Fee in Gwei (BaseGasFee+MaxPriority)*LimitGaz: {FrontRuningTxMaxFee}')
    if (difBack/int(limit_gas))<-1: # check if diff in GasPrice of Back Runing Tx and TargetedTx is less than -1 Gwei, to handle a huge gap between Tx
         BackRuningTxMaxFee=BackRuningTxMaxFee-difBack-0.065*int(limit_gas)
         BackPriorityFeeperUnitGaz=PredictedBack.reshape(-1)[0]-difBack/int(limit_gas)-0.065
         st.success(f'The transaction is most likely to be included in the same block of targeted Tx with 99% and after TargetedTx directly, with probability more than 85%: {st.session_state.TargetedBlock} after Targeted Transaction ✔️👌 / BackRuningTx: LimitGaz: {limit_gas} ; GasPrice in Gwei: {BackPriorityFeeperUnitGaz+base_gas_fee} ; max Total Fee in Gwei (BaseGasFee+MaxPriority)*LimitGaz: {BackRuningTxMaxFee}')
    elif difBack>=0: # check if back running tx max fee more than targeted tx max fee
      DeletedGasFee=CheckingBack(difBack,int(limit_gas))
      BackRuningTxMaxFee=BackRuningTxMaxFee-DeletedGasFee
      BackPriorityFeeperUnitGaz=PredictedBack.reshape(-1)[0]-(DeletedGasFee/int(limit_gas))
      st.success(f'The transaction is most likely to be included in the same block of targeted Tx with 99% and after TargetedTx directly, with probability more than 85%: {st.session_state.TargetedBlock} after Targeted Transaction ✔️👌 / BackRuningTx: LimitGaz: {limit_gas} ; GasPrice in Gwei: {(BackPriorityFeeperUnitGaz+base_gas_fee)} ; max Total Fee in Gwei (BaseGasFee+MaxPriority)*LimitGaz: {BackRuningTxMaxFee}')
    else:
      st.success(f'The transaction is most likely to be included in the same block of targeted Tx with 99% and after TargetedTx directly, with probability more than 85%: {st.session_state.TargetedBlock} after Targeted Transaction ✔️👌 / BackRuningTx: LimitGaz: {limit_gas} ; GasPrice in Gwei: {PredictedBack.reshape(-1)[0]+base_gas_fee} ; max Total Fee in Gwei (BaseGasFee+MaxPriority)*LimitGaz: {BackRuningTxMaxFee}')

#######################
