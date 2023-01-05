import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf
from sklearn import ensemble as es
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import sys

def DataLoad(TargetS,INL):
	DF = yf.download(" ".join(INL), period='5y')
	DF.fillna(method = 'ffill', inplace = True)	
	DF.to_pickle(_DFfname_)
	return DF

def RecentDF(TargetS,INL,dfmedianDF):
	DF = yf.download(" ".join(INL), period='30d')
	DF.fillna(method = 'ffill', inplace = True)	
	#DF.to_pickle(_DFfname_)
	#print(DF.tail())
	DF2 = (DF['Adj Close']/dfmedianDF)[DF.index.isocalendar().day < 6 ]
	DF2["t1"] = DF2.shift(-1)[TargetS]
	DF2["t2"] = DF2.shift(-2)[TargetS]
	DF2["t3"] = DF2.shift(-3)[TargetS]
	#print(DF2.tail())
	DF3 = DF2.dropna()

	#return DF3.iloc[-3:-2, :]
	return DF3
	
def RecentDFa(TargetS,INL,dfmedianDF):
	#DF = yf.download(" ".join(INL), start="2022-09-1", end="2022-10-04")
	DF = yf.download(" ".join(INL), period='3mo')
	DF.fillna(method = 'ffill', inplace = True)	
	#DF.to_pickle(_DFfname_)
	#print(DF.tail())
	DF2 = (DF['Adj Close']/dfmedianDF)[DF.index.isocalendar().day < 6 ]
	DF2["t1"] = DF2.shift(-1)[TargetS]
	DF2["t2"] = DF2.shift(-2)[TargetS]
	DF2["t3"] = DF2.shift(-3)[TargetS]
	#print(DF2.tail())
	DF3 = DF2.dropna()

	#return DF3.iloc[-3:-2, :]
	return [DF3, DF['Adj Close']]
	
def RecentDF2(TargetS,INL,dfmedianDF):
	DF = yf.download(" ".join(INL), period='30d')
	DF.fillna(method = 'ffill', inplace = True)	
	#DF.to_pickle(_DFfname_)
	#print(DF.tail())
	DF2 = (DF['Adj Close']/dfmedianDF)[DF.index.isocalendar().day < 6 ]
	#DF2["t1"] = DF2.shift(1)[TargetS]
	#DF2["t2"] = DF2.shift(2)[TargetS]
	#DF2["t3"] = DF2.shift(3)[TargetS]
	#print(DF2.tail())
	DF3 = DF2.dropna()

	#return DF3.iloc[-3:-2, :]
	return DF3
	
#def modelbuild(TargetS,InputSL,DF,DaysI):
def modelbuild(DF,TargetS,FeatureL,DaysI):

	XYL = modelInput(DF,TargetS,FeatureL,DaysI)

	x = XYL[0]
	y = XYL[1]
	dfmedianDF = XYL[2]

	xlenI = len(x)
	x_train = x.iloc[:int(xlenI*0.75) , :]
	x_test = x.iloc[int(xlenI*0.75): , :]

	y_train = y.iloc[:int(xlenI*0.75)]
	y_test = y.iloc[int(xlenI*0.75):]

	model = es.RandomForestRegressor()
	model.fit(x_train,y_train)
	Score = model.score(x_test,y_test)
	model2 = es.RandomForestRegressor()
	model2.fit(x,y)
	return [model, Score, x, y, dfmedianDF, model2]

def dayShift(DF,DaysNumI):
	DF2shift =DF.shift(DaysNumI)
	DF2shift.columns =  [str(x)+"t"+str(DaysNumI) for x in DF.keys()] 
	#print(DF2shift)
	return(DF2shift)

def modelInput(DF0,TargetS,FeatureL,DaysI,dfmedianDFI=1):
	DF0 = DF0.drop("Volume", axis=1)
	namesL = [ x[0].replace(" ","")+"-"+x[1] for x in list(DF0) if x[0] ]
	DF = DF0.copy()
	DF.columns = namesL
	
	if type(dfmedianDFI) == int:
		dfmedianDF = DF.median()
	else:
		dfmedianDF = dfmedianDFI

	DF2 = (DF/dfmedianDF)
	DF2 = DF2.fillna(method = 'ffill')
	#DF2 = DF2.fillna(method = 'bfill')

	#DayL = ["t"+str(i) for i in range(1,4)]
	TMPDF = pd.DataFrame() 
	TMPDF["tG"] = DF2.shift(-1*DaysI)["AdjClose-"+TargetS]
	for i in range(1,5):
		DF2t1 = dayShift(DF2,i)
		#print("type(DF2t1)")
		#print(type(DF2t1))
		TMPDF = pd.concat([ TMPDF, DF2t1 ],axis=1)

	DF2 = pd.concat([DF2,TMPDF], axis=1)
	DF2 = DF2.dropna()
	DF3 = DF2[DF2['tG'] > 0]
	DF3 = DF3.fillna(method = 'ffill')
	DF4 = DF3#[DaysI+2:]
	y = DF4.pop('tG')
	x = DF4#[InputSL + DayL ]
	return [x, y, dfmedianDF]
	
class ModelInput(object):
	def __init__(self,DataFrameDF,TargetNameS,FeatureListL,FDaysNumI=1, BDaysNumI=5, XSC=0,YSC=0,SplitNumF=0.8):
		self.DF = DataFrameDF.copy()		
		self.FDaysNumI = FDaysNumI
		self.BDaysNumI = BDaysNumI
		self.DFnamesL = [ x[0].replace(" ","")+"-"+x[1] for x in list(DataFrameDF) ]
		self.DF.columns = self.DFnamesL
		self.median = self.Median()
		self.SplitNumF= SplitNumF
		self.YSC = YSC
		self.XSC = XSC
		self.TargetNameS = "AdjClose-"+TargetNameS
		self.FeatureListL = FeatureListL
		self.DF = pd.concat([self.DF, self.Retro()], axis=1) ## Create past data Columns from previous BDaysNumI rows
		NameL = [ x for x in list(self.DF) if self.DF.median()[x] != 0 ] ## Filter out column with all None row
		self.DF = self.DF[NameL] ## Select only non-all-None Column
		self.DF = self.DF.fillna(method = 'ffill') ## Fill missing data point with previous day data
		
	def CreatePredictInput(self):
		self.X = self.DF.copy()
		self.X = self.X.dropna()
		#return self.XP
		
	def CreateTestSet(self):
		self.CreateTrainSET()
		self.testX = pd.DataFrame(self.XSC.transform(self.X))
		#self.testY = pd.DataFrame(self.YSC.transform(self.Y))
		#self.testY = self.Y
		self.testY = (self.Y).values.ravel()
	
	def CreateTrainSET(self):	
		self.X = self.DF.copy()
		self.X['TaRgEt'] =  dayShift(self.DF[self.TargetNameS], (-1 * self.FDaysNumI) )
		self.X = self.X.dropna()
		self.Y = pd.DataFrame(self.X.pop('TaRgEt'))
		#self.X = self.DF
		
		#print("self.X=\n",self.X) ##DEBUG
		if (type(self.XSC) == int):
			#self.Xscaler = self.standardize(self.X)
			self.XSC = self.standardize(self.X)
			#self.Train_X = self.Xscaler.transform(self.X)
			
		#else:
		#	self.Xscaler = self.XSC
			#self.Train_X = self.Xscaler.transform(self.X)
			
		if (type(self.YSC) == int):
			#self.Yscaler = self.standardize(self.Y)
			self.YSC = self.standardize(self.Y)
			#self.Train_Y = self.Yscaler.transform(self.Y)
		#else:
		#	self.Yscaler = self.YSC
		#print("self.X=\n",self.X) ##DEBUG
		
	def Retro(self):
		TMPDF = pd.DataFrame() 
		TMPDF["Date"] = self.DF.reset_index()['Date']
		TMPDF = TMPDF.set_index('Date')
		for i in range(1,self.BDaysNumI):
			#print("TMPDF")
			#print(TMPDF)
			DF2t1 = dayShift(self.DF,i)
			#print("DF2t1")
			#print(DF2t1)
			TMPDF = pd.concat([ TMPDF, DF2t1 ],axis=1)	
		#TMPDF = TMPDF.set_index("Date")				
		return TMPDF
		

	def Median(self): 
		Median = self.DF.median()
		return(Median)
			
	#def DFsplit(self,SplitNumF):
			
	#	xlenI = len(self.DF)
		
	#	self.x_train = self.X.iloc[:int(xlenI*SplitNumF) , :]
	#	self.x_test = self.X.iloc[int(xlenI*SplitNumF): , :]

	#	self.y_train = self.Y.iloc[:int(xlenI*SplitNumF)]
	#	self.y_test = self.Y.iloc[int(xlenI*SplitNumF):]
		
	def standardize(self, df):
		#scaler = StandardScaler()
		scaler = preprocessing.RobustScaler()
		scaler = scaler.fit(df)
		return scaler

	def BuildModel(self,ModelType='RG'):
		self.CreateTrainSET()
		SplitNumF = self.SplitNumF
		x = pd.DataFrame(self.XSC.transform(self.X))
		#y = pd.DataFrame(self.YSC.transform(self.Y))
		y = self.Y
		xlenI = len(x)
		x_train = x.iloc[:int(xlenI*SplitNumF) , :]
		x_test = x.iloc[int(xlenI*SplitNumF): , :]

		#y_train = y.iloc[:int(xlenI*SplitNumF)]
		#y_test = y.iloc[int(xlenI*SplitNumF):]
		
		y_train = (y.iloc[:int(xlenI*SplitNumF)]).values.ravel()
		y_test = (y.iloc[int(xlenI*SplitNumF):]).values.ravel()

		if ModelType=='RG':
			MD = es.RandomForestRegressor()
		elif ModelType=='ML':
			MD = MLPRegressor(random_state=1, max_iter=5000, hidden_layer_sizes=( len(self.X.keys())*2, len(self.X.keys())*2, len(self.X.keys())*2, len(self.X.keys())+1 ) )
		self.model = MD
		self.model.fit(x_train,y_train)
		self.Score = self.model.score(x_test,y_test)
		self.model1 = MD
		self.model1.fit(x,y.values.ravel())
		self.Score1 = self.model1.score(x_test,y_test)
	
def modeltest(TargetS, VL , DF0, DF1, DaysI):
	InputSL =  VL + [TargetS]
	DF0_L = modelbuild(TargetS,InputSL,DF0,DaysI)
	#print(DF0_L[2])
	#Test_X = ( DF1['Adj Close'][InputSL] )[:-1*DaysI]
	XYL = modelInput(TargetS,InputSL,DF1,DaysI)
	Test_X = XYL[0]
	#Test_X.fillna(method = 'ffill', inplace = True)
	Test_X = Test_X.dropna()
	Xmask = Test_X.index.date.astype('str')
	#Test_X = Test_X[Xmask]
	#Test_X.fillna(method = 'bfill', inplace = True)
	#Test_Y = (DF1['Adj Close'][TargetS].shift(-1*DaysI) )
	Test_Y = XYL[1]
	Test_Y =Test_Y[Xmask]
	#Test_Y.fillna(method = 'bfill', inplace = True)
	#Test_Y.fillna(method = 'ffill', inplace = True)
	#print("Test_X",Test_X)
	#print("Test_Y",Test_Y)
	
	model = DF0_L[-1]
	#print("XXX ->>>>")
	#print(Test_X/DF0_L[4][InputSL])
	#TestTFx= Test_X/DF0_L[4][InputSL]
	#TestTFy= Test_Y/DF0_L[4][TargetS]
	#print("TestTFx",TestTFx)
	#print("TestTFy",TestTFy)

	#Score= model.score(TestTFx[:-1*DaysI],TestTFy[:-1*DaysI])
	Score= model.score(Test_X,Test_Y)
	predictedA = model.predict(Test_X)

	CompDF = (pd.concat([Test_X.reset_index(), pd.DataFrame(predictedA, columns=['Predicted'])],axis=1)).set_index('Date')
	CompDF = CompDF.drop(VL,axis=1)
	return [Score, CompDF, DF0_L]
	

	
def modelsave(model,fname):
	pickle.dump(model, open(fname, 'wb'))
	return(fname)
	
def dfextract(ALLdf ,TickerS, StartDateS, EndDateS):
	if EndDateS == 0:
		mask = (ALLdf.index.date.astype('str') >= StartDateS)
	else:	
		mask = (ALLdf.index.date.astype('str') >= StartDateS) & (ALLdf.index.date.astype('str') <= EndDateS )
	VL = [ 'Open', 'Low','High',  'Close', 'Adj Close', 'Volume']
	#VTL = [[[x][TickerS]] for x in VL]
	MSKdf = ALLdf[mask]
	NewDF = pd.DataFrame()
	for i in VL: 
		NewDF[i] = MSKdf[i][TickerS]
	
	return NewDF
		
		
def futureTest(TargetS, StartDateS, EndDateS):
	Price = 10
	return Price
	
def PriceVolPlot(TargetS,DF,S="MD"):
	if S == "MA":
		TargetS="SYMC";ax=(DF["Adj Close"][TargetS+".BK"]/DF["Adj Close"][TargetS+".BK"].max()).plot();(DF["Volume"][TargetS+".BK"]/DF["Volume"][TargetS+".BK"].max()).plot(ax=ax);plt.legend(['Price','Volume']);plt.show()
	else:
		TargetS="SYMC";ax=(DF["Adj Close"][TargetS+".BK"]/DF["Adj Close"][TargetS+".BK"].median()).plot();(DF["Volume"][TargetS+".BK"]/DF["Volume"][TargetS+".BK"].max()).plot(ax=ax);plt.legend(['Price','Volume']);plt.show()

def comparePlot():
	ax = ((pd.concat([(pd.concat([NewDFL[0]['AdjClose-TTB.BK'],NewDFL[1]],axis=1).reset_index()), pd.DataFrame(Pred, columns=['Predicted']) ], axis=1)).set_index('Date'))*modelL[4]['AdjClose-TTB.BK']
	return(ax)

def main0(TargetS, INL, DaysI, DF):	

	#INL = ['EA',TargetS]
	fname = TargetS.replace(".","_") + "RegMod"
	_DFfname_ = TargetS.replace(".","_") + "_PDDF.pkl"

	## Toggle Between Newly Load and Reuse
	#DF = DataLoad(TargetS,INL) #To download new dataset
	#DF = pd.read_pickle(_DFfname_) #To use loaded dataset
	#print(DF.head())
	##
	DaysI = 1
	modelL = modelbuild(TargetS,DF,DaysI)
	#modelsave(modelL[0],fname)
	print(modelL[1])
	print(modelL[2])
	dfmedianDF = modelL[4]
	RDF = RecentDFa(TargetS,INL,dfmedianDF)
	print("RDF=")
	print(RDF[0].head())
	ResDF = pd.DataFrame(RDF[1][ RDF[1].index.isin(RDF[0].index) ][TargetS]).shift(-DaysI)
	#RDF2 = RDF.drop(TargetS, axis=1)
	PartialM = modelL[0].predict(RDF[0])
	FullM = modelL[5].predict(RDF[0])
	ResDF['Current'] = pd.DataFrame(RDF[1][ RDF[1].index.isin(RDF[0].index) ][TargetS]).shift(-DaysI)
	ResDF['PartialM'] =  PartialM
	ResDF['FullM'] =  FullM
	ResDF['Diff_PT'] =  100*(ResDF['PartialM'] - ResDF[TargetS])/ResDF['PartialM']
	ResDF['Diff_FL'] =  100*(ResDF['FullM'] - ResDF[TargetS])/ResDF['FullM']

	print(ResDF)

	#print(modelL[-1].predict(RDF))
	PreD = pd.DataFrame(ResDF[TargetS])
	#print(PreD.head())
	#print(type(PreD))

	#PreD["pred"] = modelL[0].predict(modelL[2].iloc[int(len(modelL[2])*0.75):])
	#PreD["pred2"] = modelL[-1].predict(modelL[2].iloc[int(len(modelL[2])*0.75):])
	PreD["pred"] = modelL[0].predict(RDF[0])
	PreD["pred2"] = modelL[-1].predict(RDF[0])


	#print(PreD.head())
	#print("####")
	#print("modelL[2]")
	#print(modelL[2])
	PreD.plot()
	plt.legend(["Actu","Predict0","Predict1"])
	plt.show()


def main(DF,LastMonDF,TargetS,FeatureL,DaysI):

	TmpDF = pd.DataFrame()
	MD = modelInput(LastMonDF,TargetS,FeatureL,0,1)[2]
	NewDF = modelInput(LastMonDF,TargetS,FeatureL,0,MD)
	for dI in range(1,DaysI+1):	
		modelL = modelbuild(DF,TargetS,FeatureL,dI)
		NewDFL = modelInput(LastMonDF,TargetS,FeatureL,dI,modelL[4])
		Pred = modelL[-1].predict(NewDF[0])
	#print(Pred)
		Score = modelL[-1].score(NewDFL[0],NewDFL[1])
		TmpDF = pd.concat([TmpDF,
				pd.DataFrame(Pred, columns=['Predicted'+str(dI)])#.shift(1*dI) 
			], axis=1)
		#print(TmpDF)
		print(Score)

	#comDF = ((pd.concat([(pd.concat([NewDFL[0]["AdjClose-"+TargetS],NewDFL[1]],axis=1).reset_index()), pd.DataFrame(Pred, columns=['Predicted']).shift(-1*DaysI) ], axis=1)).set_index('Date'))*modelL[4]["AdjClose-"+TargetS]
	comDF = ((pd.concat([(pd.concat([NewDF[0]["AdjClose-"+TargetS],NewDF[1]],axis=1).reset_index()),
		 TmpDF ], axis=1)).set_index('Date'))*MD["AdjClose-"+TargetS]
	print(TmpDF*modelL[4]["AdjClose-"+TargetS])
	return comDF

def THSETfound(TargetS,VL,ALLdf,FDaysNumI):
	FeatureL = VL + [TargetS]
	NN = [ x for x in list(ALLdf) if x[1] in FeatureL ]
	DF = ALLdf[NN][-1500:]
	DFlenI = len(DF)
	DFtmpL = [x for x in list(DF) if DF[x].isna().sum() < DFlenI*0.5 ]
	DF = DF[DFtmpL]
	
	DF = DF[DF['Adj Close'][TargetS] > 0 ] ## Filter Market Close Days Out
	DF0 = DF[:-50]
	inputOBJ0 = ModelInput(DF0,TargetS,FeatureL,FDaysNumI=FDaysNumI)
	#inputOBJ0.BuildModel()

	inputOBJ0.BuildModel(ModelType='ML')
	MLmodel1 = inputOBJ0.model1
	MLmodel1ScoreI = inputOBJ0.Score1

	inputOBJ0.BuildModel(ModelType='RG')
	RGmodel1 = inputOBJ0.model1
	RGmodel1ScoreI = inputOBJ0.Score1
	
	return([MLmodel1ScoreI,MLmodel1,RGmodel1ScoreI,RGmodel1 ])
	
def ModLoop(DF0,TargetS,FeatureL,FDaysNumIcount=1):
	ModLoopOutL = []
	for i in range(1,FDaysNumIcount+1):
		inputOBJ0 = ModelInput(DF0,TargetS,FeatureL,FDaysNumI= i )

		inputOBJ0.BuildModel(ModelType='RG')
		MLmodel1 = inputOBJ0.model1
		MLmodel1ScoreI = inputOBJ0.Score1
		
		ModLoopOutL.append([str(FDaysNumI) + "Day",TargetS, MLmodel1ScoreI," ".join(FeatureL), MLmodel1])
			
	return(ModLoopOutL)
	

def ModOpt(TargetS,VL0,ALLdf,FDaysNumI):
	ModScoreL = []
	MaxF = 0.0
	VL = []
	for v in VL0:
		VL = VL + [v] 
		try:
			FeatureL = VL + [TargetS]
			NN = [ x for x in list(ALLdf) if x[1] in FeatureL ]
			DF = ALLdf[NN][:]
			DFlenI = len(DF)
			DFtmpL = [x for x in list(DF) if DF[x].isna().sum() < DFlenI*0.5 ]
			DF = DF[DFtmpL]
			
			DF = DF[DF['Adj Close'][TargetS] > 0 ] ## Filter Market Close Days Out
			DF0 = DF[:-50]
			inputOBJ0 = ModelInput(DF0,TargetS,FeatureL,FDaysNumI=FDaysNumI)

			inputOBJ0.BuildModel(ModelType='RG')
			MLmodel1 = inputOBJ0.model1
			MLmodel1ScoreI = inputOBJ0.Score1
			
			#print(str([" ".join(VL), MLmodel1ScoreI]))
			if MLmodel1ScoreI > MaxF:
				MaxF = MLmodel1ScoreI
				model = MLmodel1
			else:
				VL.remove(v)
			ModScoreL = [str(FDaysNumI) + "Day",TargetS," ".join(VL), MaxF, model]	
		except:
			print("ERROR at ",v)
	
	return(ModScoreL)


"""
#TargetS = "DELTA.BK"
INL = ['^SET.BK', 'THB=X', 'GC=F', 'CL=F', 'ETH-USD', 'GBP=X', 'EUR=X', 'JPY=X', 'HG=F', TargetS]
"""
BigTHL = ".BK ".join("AOT MINT PTTEP BGRIM GULF KBANK PTT BANPU BH BDMS PTTGC CPALL ADVANC BBL MTC TOP CRC KTB CPN SCC EA GPSC SCB TIDLOR SAWAD CBG KCE IVL JMT SCGP BLA HMPRO OR LH BEM JMART TISCO AWC TU OSP KTC CPF TTB DTAC INTUCH IRPC GLOBAL TRUE EGCO BTS.BK".split()).split()
FutureL = "=F ".join("CL HO NG RB BZ ZC ZO KE ZR ZM ZS GF HE LE KC CT OJ SB LBS=F".split()).split()
CryptoL = [ 'BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'USDT-USD']
WorldSetL = ["^GSPC","^TNX",'^SET.BK', "^N225", "^HSI", "^KS11", "^N100"]
MetalL = ["SI=F", "HG=F", 'GC=F', "ALI=F", "PL=F", "PA=F", ]
FiatL = ['THB=X','GBP=X', 'EUR=X', 'JPY=X',"CNH=X","RUB=X","VND=X", "AUD=X", "INR=X", "MYR=X", "IDR=X", "PHP=X", "SGD=X", "PKR=X", "MMK=X", "KRW=X",  "CAD=X" ]

ALL = WorldSetL + FutureL + CryptoL + MetalL + FiatL  + BigTHL
#ALLdf 
#mask = (ALLdf.index.date.astype('str') >= "2017-01-01") & (ALLdf.index.date.astype('str') < "2022-01-01" )

"""
##
"""
ALLdf = pd.read_pickle('DF5y.pkl')
#AllTHAIDF = pd.read_pickle('AllTHAIDF2017-22.pkl')
#mask = (ALLdf.index.date.astype('str') >= "2021-01-01") & (ALLdf.index.date.astype('str') < "2022-01-01" )
#mask2 = (ALLdf.index.date.astype('str') >= "2021-03-01") & (ALLdf.index.date.astype('str') < "2022-03-30" ) 


#print("DF0",DF0)
#DF1 = ALLdf[mask2]
#print("DF1",DF1)
TargetS =  "PTT.BK"
FDaysNumI = 1

if len(sys.argv) >= 2:
	TargetS = sys.argv[1] + ".BK"
if len(sys.argv) == 3:
	FDaysNumI = int(sys.argv[2])
#VL = ['CL=F','^SET.BK','GC=F','THB=X']
VL = ['^SET.BK','THB=X', 'GC=F', 'CL=F', 'ETH-USD', 'GBP=X', 'EUR=X', 'JPY=X', 'HG=F'] #+ WorldSetL 
#VL = ['THB=X'] + WorldSetL # + FutureL
#VL = ['^SET.BK','GC=F','CL=F']
#BigTHL.remove('TTB.BK')
#VL = ['^SET.BK','THB=X','GC=F','CL=F'] #+ BigTHL
#VL =  ['^SET.BK']

#DD = pd.DataFrame(ALLdf.index[-40:])
#DD['D'] = DD.index * 0 + (ALLdf['Adj Close', TargetS]).median()
#DD = DD.set_index(['Date'])
#ax = DD.plot()

#CorSet = "PB.BK,GC=F,PA=F,TFMAMA.BK,INR=X,AIMIRT.BK,CBG.BK,GULF.BK,PKR=X,MYR=X,AUCT.BK,TQM.BK,BTC-USD,BLAND.BK,LH.BK,TOP.BK,S.BK,STANLY.BK,IRPC.BK,BBL.BK,PLE.BK,SCCC.BK,PTTGC.BK".split(",")
#IRPCCorSet = "INR=X,AUD=X,MYR=X,PKR=X,KRW=X,^TNX,^HSI,^SET.BK,ZR=F".split(",")
#CorSet=IRPCCorSet

VL = ['^SET.BK','THB=X', 'GC=F', 'CL=F', 'ETH-USD', 'GBP=X', 'EUR=X', 'JPY=X', 'HG=F']
modelsPoolL = []


for i in BigTHL[:0]:
	modelL = THSETfound(i,VL,ALLdf,FDaysNumI)
	print(i)
	print("ML:",modelL[0])
	print("RG:",modelL[2])
	if modelL[0] > 0.95 or modelL[2] > 0.95 :
		modelsPoolL.append({i:modelL})
		
for i in BigTHL[:]: ## Predict price of next 5 days
	print(ModOpt(i,VL,ALLdf,1))
	print(ModOpt(i,VL,ALLdf,2))
	print(ModOpt(i,VL,ALLdf,3))
	print(ModOpt(i,VL,ALLdf,4))
	print(ModOpt(i,VL,ALLdf,5))

for i in WorldSetL[:0]:
	

	FeatureL = VL + [TargetS] + [i] + CorSet
	
	#DaysI = 1
	NN = [ x for x in list(ALLdf) if x[1] in FeatureL ]
	#TN = [ x for x in list(AllTHAIDF) if x[1] in FeatureL and  x[1] not in NN]
	#N2 = NN + TN 
	#print(FeatureL)
	#print(N2)

	#DF = ALLdf[NN]
	#NEWDF = pd.concat([ALLdf,AllTHAIDF[TN]], axis=1)
	DF = ALLdf[NN][-1500:]
	DFlenI = len(DF)
	DFtmpL = [x for x in list(DF) if DF[x].isna().sum() < DFlenI*0.5 ]
	DF = DF[DFtmpL]
	
	#NNN = 
	#DF = DF[DF.index.isocalendar().day < 6 ] ## Filter Market Days only
	DF = DF[DF['Adj Close'][TargetS] > 0 ] ## Filter Market Close Days Out
	DF0 = DF[:-50]
	#LastMonDF = DF[-30:]

 
	inputOBJ0 = ModelInput(DF0,TargetS,FeatureL,FDaysNumI=FDaysNumI)
	inputOBJ0.BuildModel()
	#RGmodel0 = inputOBJ0.model
	#RGmodel1 = inputOBJ0.model1
	#print(inputOBJ0.Score)
	#print(inputOBJ0.Score1)

	inputOBJ0.BuildModel(ModelType='RG')
	MLmodel0 = inputOBJ0.model
	MLmodel1 = inputOBJ0.model1
	print(inputOBJ0.Score)
	print(inputOBJ0.Score1)

	Recent2moDF = LastMonDF = DF[-40:]
	Recent2moDFOBJ = ModelInput(Recent2moDF,TargetS,FeatureL,XSC=inputOBJ0.XSC,YSC=inputOBJ0.YSC,FDaysNumI=FDaysNumI)
	#Recent2moDFOBJ.CreatePredictInput()
	Recent2moDFOBJ.CreateTestSet()
	#print("Recent2moDF test RGscore")
	#print(Recent2moDFOBJ.testX)
	#print(Recent2moDFOBJ.testY)
	#RGscore = RGmodel1.score(Recent2moDFOBJ.testX, Recent2moDFOBJ.testY)
	#print(RGscore)

	print("Recent2moDF test MLscore")
	MLscore = MLmodel1.score(Recent2moDFOBJ.testX, Recent2moDFOBJ.testY)
	print(FeatureL) 
	print("MLscore=",MLscore)
	Recent2moDFOBJ.CreatePredictInput()
	MLpredicted = pd.DataFrame(MLmodel1.predict( inputOBJ0.XSC.transform(Recent2moDFOBJ.X) ), columns=['MLpredicted'] )
	DateL = Recent2moDFOBJ.X.index.to_series()
	ComDF = pd.DataFrame(Recent2moDFOBJ.X[Recent2moDFOBJ.TargetNameS][DateL])
	ComDF = ComDF.set_index(DateL)
	MLpredictedDF = pd.DataFrame(MLpredicted).set_index(DateL)
	#ComDF['MLpredicted'+str(i)+'D'] = MLpredictedDF.shift(FDaysNumI)
	ComDF['MLpredicted'] = MLpredictedDF
	ComDF.plot(ax=ax)
#plt.show()

#Recent2moDF =  yf.download(" ".join(FeatureL), period='3mo' )
#Recent2moDF.to_pickle("Recent2moDF.pkl")
#Recent2moDF = pd.read_pickle("Recent2moDF.pkl")
"""
Recent2moDFOBJ = ModelInput(Recent2moDF,TargetS,FeatureL,XSC=inputOBJ0.XSC,YSC=inputOBJ0.YSC,FDaysNumI=FDaysNumI)
Recent2moDFOBJ.CreatePredictInput()
MLpredicted = pd.DataFrame(MLmodel1.predict( inputOBJ0.XSC.transform(Recent2moDFOBJ.X) ), columns=['MLpredicted'] )
MLpredictedDF = pd.DataFrame(MLpredicted)
"""
###
"""
Recent2moDF = LastMonDF = DF[-40:]
Recent2moDFOBJ = ModelInput(Recent2moDF,TargetS,FeatureL,XSC=inputOBJ0.XSC,YSC=inputOBJ0.YSC,FDaysNumI=FDaysNumI)
#Recent2moDFOBJ.CreatePredictInput()
Recent2moDFOBJ.CreateTestSet()
#print("Recent2moDF test RGscore")
#print(Recent2moDFOBJ.testX)
#print(Recent2moDFOBJ.testY)
#RGscore = RGmodel1.score(Recent2moDFOBJ.testX, Recent2moDFOBJ.testY)
#print(RGscore)

print("Recent2moDF test MLscore")
MLscore = MLmodel1.score(Recent2moDFOBJ.testX, Recent2moDFOBJ.testY)
print(MLscore)

Recent2moDFOBJ.CreatePredictInput()
MLpredicted = inputOBJ0.YSC.inverse_transform( pd.DataFrame(MLmodel1.predict( inputOBJ0.XSC.transform(Recent2moDFOBJ.X) ), columns=['MLpredicted'] ) )
#DateL = Recent2moDFOBJ.X.index.date.astype('str')
#DateL = Recent2moDFOBJ.X.index.date.astype('str')
DateL = Recent2moDFOBJ.X.index.to_series()
ComDF = pd.DataFrame(Recent2moDFOBJ.X[Recent2moDFOBJ.TargetNameS][DateL])
ComDF = ComDF.set_index(DateL)
MLpredictedDF = pd.DataFrame(MLpredicted).set_index(DateL)
ComDF['MLpredicted'+str(FDaysNumI)+'D'] = MLpredictedDF.shift(FDaysNumI)
ComDF.plot()
plt.show()
"""
###





##########
"""

RGpredicted = inputOBJ0.YSC.inverse_transform( pd.DataFrame(RGmodel1.predict( inputOBJ0.XSC.transform(Recent2moDFOBJ.X) ), columns=['RGpredicted'] ) )
print(RGpredicted)
MLpredicted = inputOBJ0.YSC.inverse_transform( pd.DataFrame(MLmodel1.predict( inputOBJ0.XSC.transform(Recent2moDFOBJ.X) ), columns=['MLpredicted'] ) )

DateL = Recent2moDFOBJ.X.index.date.astype('str')
ComDF = pd.DataFrame(Recent2moDFOBJ.X[Recent2moDFOBJ.TargetNameS][DateL])
ComDF = ComDF.set_index(DateL)
RGpredictedDF = pd.DataFrame(RGpredicted).set_index(DateL)
ComDF['RGpredicted'+str(FDaysNumI)+'D'] = RGpredictedDF.shift(FDaysNumI)
MLpredictedDF = pd.DataFrame(MLpredicted).set_index(DateL)
ComDF['MLpredicted'+str(FDaysNumI)+'D'] = MLpredictedDF.shift(FDaysNumI)
#ComDF['MLDiff'] =  ComDF['MLpredicted1D'] - ComDF[Recent2moDFOBJ.TargetNameS]
#ComDF['RGDiff'] =  ComDF['RGpredicted1D'] - ComDF[Recent2moDFOBJ.TargetNameS]
ComDF.plot()
plt.show()

Recent2moDFOBJ.CreatePredictInput()
DateL = Recent2moDFOBJ.X.index.date.astype('str')
ComDF = pd.DataFrame(Recent2moDFOBJ.X[Recent2moDFOBJ.TargetNameS][DateL])
ComDF = ComDF.set_index(DateL)

RGpredicted = inputOBJ0.YSC.inverse_transform( pd.DataFrame(RGmodel1.predict( inputOBJ0.XSC.transform(Recent2moDFOBJ.X) ), columns=['RGpredicted'] ) )
print(RGpredicted)
MLpredicted = inputOBJ0.YSC.inverse_transform( pd.DataFrame(MLmodel1.predict( inputOBJ0.XSC.transform(Recent2moDFOBJ.X) ), columns=['MLpredicted'] ) )
RGpredictedDF = pd.DataFrame(RGpredicted).set_index(DateL)
ComDF['RGpredicted1D'] = RGpredictedDF.shift(FDaysNumI)
MLpredictedDF = pd.DataFrame(MLpredicted).set_index(DateL)
ComDF['MLpredicted1D'] = MLpredictedDF.shift(FDaysNumI)
"""
###################


