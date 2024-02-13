import streamlit as st
import os
############################# Config page ############
st.set_page_config(
    page_title="Home",
    page_icon="static/icon.png",  
)
###########################################

st.title("User Guide: How to Use This App💡")
st.write(""" Here, you will find documentation and a comprehensive guide on how to use this application to predict Gas Fees and Block Numbers for transactions in the Ethereum blockchain, specifically for the MEV (Miner Extractable Value) sandwich attack strategy. """)
st.markdown(
    """
    ### 💾 The Guide:
    To develop a system capable of predicting Front-running and Back-running transactions MaxPriority or Priority, we utilized deep learning models such as LSTM for forecasting GasBasefee for upcoming blocks and MLP for predicting maximum priority gas fees.

    For training the LSTM model, we leveraged a dataset comprising over 40,000 observations, and for testing, we utilized more than 16,000 observations. Each observation contained information about the block, including BlockTimeStamp, GasBaseFees, Gas used, Block size, etc. We incorporated a lag of 1 timestamp as we found it to be optimal for model performance. Our choice of loss function was the mean squared error.

    The LSTM model demonstrated strong performance in predicting BaseGasFee (in Gwei). Notably, it exhibited minimal deviation in errors between real BaseGasFee and predicted values, even when tested on new data. The consistency in error values between the training and testing datasets, with minimal dispersion, further underscores the model's reliability. These results affirm the efficacy of our LSTM model for Block GasBaseFee prediction.
    
    | Data | MSE    | MAE | MAX Error | 75% percentile of errors distribution | 95% percentile of errors distribution |
    |---------|-----|----------|---------|-----|-----|
    | Train   | 0.0000575 Gwei|0.00506 Gwei |0.079949 Gwei|0.002332 Gwei|0.006 Gwei|
    | Test     | 0.0000571 Gwei  | 0.00503 Gwei |0.110665 Gwei|0.002435 Gwei|0.0061 Gwei|

    From these amazing results, we can conclude that 95% of the predicted values exhibit a difference from the real value of BaseGasFee of less than 0.0061 Gwei. In the worst-case scenario, the error is just 0.11 Gwei.
    Note: We apply our model to predict the GasBaseFee of the next second, third, fourth, and fifth blocks to address the challenge posed by the short time span of 12 seconds between two consecutive blocks, which can make executing two transactions within this timeframe difficult. The model demonstrates good performance; however, as expected, its accuracy in predicting the GasBaseFee of the next block surpasses that of subsequent blocks. For 95% of cases, the errors in predicting the base gas fee of the second, third, fourth, and fifth next blocks, respectively, are less than the following values: 1.60 Gwei, 1.95 Gwei, 2.21 Gwei, and 2.42 Gwei. Nonetheless, it's important to note that there is a 1% probability where errors can reach the following maximum values for each prediction: 7 Gwei, 13 Gwei, 14 Gwei, and 16 Gwei, indicating a notable dispersion of errors. The following table presents all related results:
    
    | Block| MSE    | MAE | 75% percentile of errors distribution | 95% percentile of errors distribution | 99% percentile of errors distribution | MAX Error |
    |-----|---------|-----|----------|---------|-----|-----|
    | Second Next Block|0.815 Gwei |0.623 Gwei|0.33 Gwei|1.60 Gwei|3.01 Gwei|7.06 Gwei|
    | Third Next Block  | 1.360 Gwei |0.800 Gwei|0.47 Gwei|1.95 Gwei|3.88 Gwei| 13.60 Gwei|
    | Fourth Next Block|1.810 Gwei |0.920 Gwei|0.58 Gwei|2.21 Gwei|4.37 Gwei| 14.70 Gwei|
    | Five Next Block  | 2.250 Gwei |1.025 Gwei|0.64 Gwei|2.42 Gwei|4.83 Gwei|16.93 Gwei|
    -------------------------------------------------------------------------------------
    Regarding the MLP model utilized for predicting the maximum priority gas fee (in Gwei) for front-running and back-running transactions, our training dataset comprised approximately one million observations. Each observation contained information about the targeted transaction, such as MaxPriorityGasFee, LimitGas, and GasBase. For testing, we utilized over 321,000 transaction observations.

    The results demonstrate consistency between the model's performance on known data (TrainData) and unseen data (TestData). However, a notable issue arises concerning the dispersion of errors in predicting MaxPriorityGasFees for front-running and back-running transactions. To address this concern, we employed descriptive statistics for front-running and back-running transactions data, including mean, standard deviation, median, 75th percentile, 95th percentile, and 99th percentile for  each block.

    By leveraging these statistics, we gained insights into the distribution of MaxPriorityGasFees. We then utilized this information to mitigate the dispersion of modeling errors. Specifically, we applied conditions based on the transaction types: front-running and back-running. If the predicted MaxPriorityGasFee for a front-running transaction was less than the targeted transaction fees, we adjusted the prediction by adding the value of the 99th percentile or mean, This adjustment ensured that the predicted FrontRunningTxFees were greater than the TargetedTransactionTxFees. Similarly, for back-running transactions, if the predicted fee was greater than the targeted transaction fees, we made adjustments accordingly (All that to ensure this following placement of transactions within the block: FrontRuningTx -> TargetedTx -> BackRuningTx).
    
    The following results outline the performance of the MLP model in predicting MaxPriorityGasFees:
    - FrontRuningTransactions :
    ---------------------------------------------------
    | Data | MSE    | MAE | MAX Error | 95% percentile of errors distribution | 99% percentile of errors distribution |
    |---------|-----|----------|---------|-----|-----|
    | Train   | 77.47 Gwei|1.45 Gwei |300.29 Gwei|0.96 Gwei|14.96 Gwei|
    | Test     | 74.76 Gwei  | 1.44 Gwei |300.58 Gwei|0.96 Gwei|14.71 Gwei|

    - BackRuningTransactions :
    ---------------------------------------------------
    | Data | MSE    | MAE | MAX Error | 95% percentile of errors distribution | 99% percentile of errors distribution |
    |---------|-----|----------|---------|-----|-----|
    | Train   | 63.64 Gwei|2.78 Gwei |297.42 Gwei|-0.71 Gwei|12.44 Gwei|
    | Test     | 62.78 Gwei  | 2.77 Gwei |297.24 Gwei|-0.73 Gwei|12.30 Gwei|
    
    From these results, we can conclude that for Front-running transactions, 99% of prediction errors are less than 14.71 Gwei, with 95% of errors being less than 0.96 Gwei. While this is generally acceptable, the dispersion of errors occasionally leads to outliers, with rare errors exceeding 200 Gwei or falling below -200 Gwei, occurring between 1% and 2% of the time.
    . Also, we can conclude that for Back-running transactions, 99% of prediction errors are less than 12.44 Gwei, with 95% of errors being less than -0.73 Gwei. Probability of rare errors is between 1% - 2%.
    
    My approach to predicting the block number in which a specific transaction in the mempool will be registered is straightforward. I estimate the confirmation time of the targeted transaction and leverage the fact that, in the majority of cases, Ethereum blocks are created approximately every 12 seconds. By dividing the confirmation time of the transaction by 12 and adding the result to the number of the most recent block, I obtain the final prediction of the estimated block number where the targeted transaction will be registered.

    This approach is simple and reliable because it relies on the Etherscan API (the most famous and reliable Block Explorer and Analytics Platform for Ethereum), to estimate the confirmation time of the targeted transaction. Additionally, it is based on the common understanding that Ethereum blocks are created every 12 seconds. Moreover, this approach is robust and less complex compared to using sophisticated modeling techniques that require extensive data collection, which can be time-consuming.
""")
st.markdown("""
            ### How to use this app?
            """)
st.markdown("""
        ### 📋 Predict GasBaseFees for next Block:
         - Input this data about Last Stacked or Mined Block in Ethereum Blockchain : 'BlockTimeStamp','BlockNumber','GasUsed','BlockSize','NbrTx','BaseGasFee'.
         - Click on Predict GasBaseFees Button.
        ### 📋 Predict MaxPriorityGasFee for Front-runing and Back-runing Transaction:
         - Input this data about Targeted Transaction : 'MaxPriority','GasBaseTxFee','LimitGaz' 
         - Click on Predict MaxPriority for Front and back runing transactions.
        ### ⚠️ Warning:
         - Please don't forget that: To predict GaseBaseFees for next Block Just click on Predict GasBaseFees Button, Inputs will be entred automatically by using EtherScan API.
         - To predict MaxPriority for Front-runing and Back-runing Transaction, you will need to input this following data of Targeted Transaction: 'MaxPriority': MaxPriority or Priority Denominated in Gwei (miner tip),'GasBaseTxFee': GaseBaseFee of the block that will register Targeted Transactions, it will be entred automaticaly based on prediction of GaseBaseFees of our LSTM model,'LimitGaz': The maximum number of gas units allowed for the transaction, not gas used ok!
        """)


st.warning("⚠️: Please pay attention to your input to ensure accurate results, and be sure to make predictions by order, don't click on buttons of predictions randomly, please respect order of button from above to buttom to get Good estimations without misleading predictions!!!")
  