(* ::Package:: *)

(* ::Title:: *)
(*Classification of images from the web*)


(* ::Subtitle:: *)
(*Basic example of downloading images from the web and doing classifications*)


(* ::Subtitle:: *)
(*Wednesday*)


(* ::Subsubtitle:: *)
(*Anton Antonov*)
(*Digital Humanities at Oxford Summer School 2017*)
(*July 2017*)


(* ::Section:: *)
(*Mission statement*)


(* ::Text:: *)
(*With this notebook we show how to obtain, prepare, and do classification experiments with a curated database of images from the Web.*)


(* ::Text:: *)
(*Secondary goal is to compare different classifiers.*)


(* ::Text::Italic:: *)
(*The section "NNMF basis" follows the Markdown document "Handwritten digits classification by matrix factorization" from MathematicaVsR at GitHub.*)


(* ::Section:: *)
(*UCI Li photograph*)


(* ::Text:: *)
(*Actually this page is better. Also see this page.*)


(* ::Input:: *)
(*dirName="/Users/yaman/Downloads/Anton-notebooks-DHOxSS-2017/li_photograph/";*)


(* ::Input:: *)
(*annotationText=ReadList[dirName<>"annotation.txt",String];*)


(* ::Input:: *)
(*ColumnForm[annotationText[[1;;4]]] (*ilk dort row*)*)


(* ::Input:: *)
(*annotations=Map[StringCases[#,n:(DigitCharacter..)~~(WhitespaceCharacter..)~~name:(Except[WhitespaceCharacter]..)~~(WhitespaceCharacter..)~~class:(___):>{n,name,class}][[1]]&,annotationText];*)
(*Dimensions[annotations]*)


(* ::Input:: *)
(*{2360,3}*)
(*annotations*)


(* ::Input:: *)
(*RecordsSummary[annotations[[All,3]]][[1]]*)


(* ::Input:: *)
(*Symbol[]*)


(* ::Input:: *)
(*imageSubDirName=dirName<>"image.cd/"*)


(* ::Input:: *)
(*"/Users/yaman/Downloads/Anton-notebooks-DHOxSS-2017/li_photograph/image.cd/"*)


(* ::Input:: *)
(*AbsoluteTiming[*)
(*images=Map[Import[imageSubDirName<>#[[2]]]&,annotations]; (*comcreateing the images. takes the second element and glues them*)*)
(*]*)


(* ::Input:: *)
(*AbsoluteTiming[*)
(*images=ConformImages[images]; (*fotolari sekilleyebiliyormusuz bu functionla*)*)
(*]*)


(* ::Input:: *)
(*Tally[ImageDimensions/@images]*)


(* ::Input:: *)
(*{{{384,256},2360}}*)


(* ::Input:: *)
(*{{{384,256},2360}}*)
(*Length[images]*)


(* ::Input:: *)
(*data=*)
(*Thread[*)
(*images->*)
(*Map[Which[*)
(*#=="flower"||#=="water plant, flower","flower",*)
(*#=="firework",#,*)
(*True,"other"*)
(*]&,annotations[[All,3]]]*)
(*]; (*BURDA HATA VERIYOR! CRASH YAPIYOR*)*)


(* ::Input:: *)
(*RandomSample[data,6]*)


(* ::Input:: *)
(*trainingInds=Flatten@Map[(pos=Flatten[Position[data[[All,2]],#]];RandomSample[pos,Round[0.7Length[pos]]])&,Union[data[[All,2]]]];*)
(*testInds=Complement[Range[Length[data]],trainingInds];*)


(* ::Input:: *)
(*Length/@{trainingInds,testInds}*)


(* ::Input:: *)
(*Tally[data[[All,-1]]]*)


(* ::Input:: *)
(*If[True,*)
(*data[[All,1]]=ConformImages[data[[All,1]],{56,56},ColorSpace->"Grayscale"];*)
(*]*)


(* ::Subsubsection:: *)
(*With a "NeuralNetwork" classifier*)


(* ::Input:: *)
(*AbsoluteTiming[*)
(*flowerFunc=Classify[data[[trainingInds]],Method->"NeuralNetwork"]*)
(*]*)


(* ::Input:: *)
(*AbsoluteTiming[*)
(*cm=ClassifierMeasurements[flowerFunc,data[[testInds]]]*)
(*]*)


(* ::Input:: *)
(*cm[{"Accuracy","Precision","Recall"}]*)


(* ::Input:: *)
(*cm[{"ROCCurve"}]*)


(* ::Input:: *)
(*cm["ConfusionMatrixPlot"]*)


(* ::Subsubsection:: *)
(*With a "NearestNeighbors" classifier*)


(* ::Input:: *)
(*AbsoluteTiming[*)
(*flowerFunc=Classify[data[[trainingInds]],Method->"NearestNeighbors"]*)
(*]*)


(* ::Input:: *)
(*AbsoluteTiming[*)
(*cm=ClassifierMeasurements[flowerFunc,data[[testInds]]]*)
(*]*)


(* ::Input:: *)
(*cm[{"Accuracy","Precision","Recall"}]*)


(* ::Input:: *)
(*cm[{"ROCCurve"}]*)


(* ::Input:: *)
(*cm["ConfusionMatrixPlot"]*)


(* ::Section:: *)
(*NNMF basis*)


(* ::Text:: *)
(*Here we follow the Markdown document "Handwritten digits classification by matrix factorization" from MathematicaVsR at GitHub.*)


(* ::Subsection:: *)
(*Getting data*)


(* ::Input:: *)
(*Reverse@Sort@Select[Tally[annotations[[All,3]]],#[[2]]>20&]*)


(* ::Input:: *)
(*data=Thread[images->Map[Which[#=="flower"||#=="water plant, flower","flower",#=="firework",#,True,"other"]&,annotations[[All,3]]]];*)


(* ::Input:: *)
(*trainImages=ConformImages[data[[trainingInds,1]],{56,56},ColorSpace->"Grayscale"];*)
(*trainImagesLabels=data[[trainingInds,2]];*)


(* ::Input:: *)
(*RandomSample[trainImages,12]*)


(* ::Input:: *)
(*RandomSample[trainImagesLabels,12]*)


(* ::Input:: *)
(*testImages=ConformImages[data[[testInds,1]],{56,56},ColorSpace->"Grayscale"];*)
(*testImagesLabels=data[[testInds,2]];*)


(* ::Subsection:: *)
(*Linear vector space representation*)


(* ::Input:: *)
(*trainImagesMat=N@SparseArray[Flatten@*ImageData/@trainImages]*)


(* ::Input:: *)
(*testImagesMat=N@SparseArray[Flatten@*ImageData/@testImages]*)


(* ::Input:: *)
(*classLabels=Union[trainImagesLabels,testImagesLabels]*)


(* ::Subsection:: *)
(*The matrix factorization*)


(* ::Input:: *)
(*nBasisSize=40;*)
(*AbsoluteTiming[*)
(*nnmfRes=ParallelMap[Function[{cl},Print[Style[cl,Bold,Red]];*)
(*pos=Flatten@Position[trainImagesLabels,cl];*)
(*tmat=trainImagesMat[[pos,All]];*)
(*res=GDCLS[tmat,nBasisSize,PrecisionGoal->4,"MaxSteps"->20,"RegularizationParameter"->0.1,"PrintProfilingInfo"->False];*)
(*bres=RightNormalizeMatrixProduct@@res;*)
(*Join[bres,{(Norm/@res[[2]])/(Norm/@bres[[2]]),PseudoInverse[bres[[2]]],tmat}]],classLabels,DistributedContexts->Automatic];*)
(*nnmfRes=AssociationThread[classLabels->nnmfRes];]*)


(* ::Input:: *)
(*Grid[ArrayReshape[#,{4,3},""]&@Map[ListPlot[nnmfRes[#][[3]],PlotRange->MinMax[Flatten@Normal@Values[nnmfRes][[All,3]]],PlotStyle->PointSize[0.02],PlotTheme->"Scientific",ImageSize->190,PlotRange->All,PlotLabel->#]&,Keys[nnmfRes]]]*)
(**)


(* ::Input:: *)
(*Magnify[#,1.6]&@Grid[Partition[#,10]&@Map[ImageAdjust@Image@Partition[#,56]&,nnmfRes["flower"][[2]]],Dividers->All,FrameStyle->GrayLevel[0.7]]*)


(* ::Subsection:: *)
(*Classification functions*)


(* ::Input:: *)
(*Clear[NNMFClassifyImageVector]*)
(*Options[NNMFClassifyImageVector]={"PositiveDifference"->False,"NumberOfNNs"->4,"WeightedNNsAverage"->False,"RepresentationNorm"->False};*)
(*NNMFClassifyImageVector[factorizationRes_Association,vec_?VectorQ,opts:OptionsPattern[]]:=Block[{residuals,invH,nW,nf,approxVec,scores,pos,rv,nnns=OptionValue["NumberOfNNs"],inds,ws},residuals=Map[(invH=factorizationRes[#][[4]];*)
(*(*nW=(#/Norm[#])&/@factorizationRes[#]\[LeftDoubleBracket]1\[RightDoubleBracket];*)nf=Nearest[factorizationRes[#][[1]]->Range[Dimensions[factorizationRes[#][[1]]][[1]]]];*)
(*If[TrueQ[OptionValue["RepresentationNorm"]],approxVec=vec.invH;*)
(*CosineDistance[Flatten[factorizationRes[#][[1]][[nf[approxVec]]]],approxVec],(*ELSE*)inds=nf[vec.invH,OptionValue["NumberOfNNs"]];*)
(*If[TrueQ[OptionValue["WeightedNNsAverage"]],ws=Map[Norm[vec-#]&,(factorizationRes[#][[5]])[[inds]]];*)
(*approxVec=ws.(factorizationRes[#][[5]])[[inds]],(*ELSE*)approxVec=Total[(factorizationRes[#][[5]])[[inds]]]];*)
(*rv=vec/Norm[vec]-approxVec/Norm[approxVec];*)
(*If[TrueQ[OptionValue["PositiveDifference"]],rv=Clip[rv,{0,\[Infinity]}]];*)
(*Norm[rv]])&,Keys[factorizationRes]];*)
(*{Keys[factorizationRes][[Position[residuals,Min[residuals]][[1,1]]]],AssociationThread[Keys[factorizationRes]->residuals]}];*)


(* ::Subsection:: *)
(*Classification*)


(* ::Input:: *)
(*AbsoluteTiming[nnmfClResInv=ParallelMap[NNMFClassifyImageVector[nnmfRes,#,"RepresentationNorm"->False,"NumberOfNNs"->30,"WeightedNNsAverage"->False]&,#&/@testImagesMat];]*)


(* ::Input:: *)
(*nnmfClResDT=Transpose[{testImagesLabels,nnmfClResInv[[All,1]]}];*)


(* ::Subsubsection:: *)
(*Total accuracy*)


(* ::Input:: *)
(*N@Mean[(Equal@@@nnmfClResDT)/.{True->1,False->0}]*)


(* ::Subsubsection:: *)
(*Precision per class*)


(* ::Input:: *)
(*t=Map[Association@Flatten@{"Label"->#[[1,1]],"NImages"->Length[#],"Precision"->N@Mean[(Equal@@@#)/.{True->1,False->0}]}&,GatherBy[nnmfClResDT,First]];*)
(*t=SortBy[t,First];*)
(*Dataset[t]*)


(* ::Subsubsection:: *)
(*ROC curve*)


(* ::Text:: *)
(*Fairly complicated to computer here!*)


(* ::Input:: *)
(*thRange=Range[0,1,0.025];*)


(* ::Input:: *)
(*aROCs=*)
(*Association@*)
(*Table[( *)
(*nonTargetClass="Non"<>targetClass;*)
(*targetClass->*)
(*Table[( *)
(*mf=Join[AssociationThread[classLabels->Table[1,Length[classLabels]]],<|targetClass->th|>];*)
(*clRes=Map[Merge[{mf,#},Times@@#&]&,nnmfClResInv[[All,2]]];*)
(*cres=Position[#,Min[#]][[1,1,1]]&/@clRes;*)
(*cres=If[#==targetClass,#,nonTargetClass]&/@cres;*)
(*ToROCAssociation[{targetClass,nonTargetClass},Map[If[#==targetClass,#,nonTargetClass]&,testImagesLabels],cres]),{th,thRange}]),*)
(*{targetClass,classLabels}*)
(*];*)


(* ::Input:: *)
(*AssociationMap[ROCPlot[thRange,aROCs[#],"PlotJoined"->Automatic,GridLines->Automatic]&,Keys[aROCs]]*)


(* ::Section::Closed:: *)
(*Getting artsy*)


(* ::Text:: *)
(*See https://mathematica.stackexchange.com/a/141277/34008 .*)


(* ::Subsection:: *)
(*Adapter*)


(* ::Input:: *)
(*Tally[#[[3]]=="flower"&/@annotations]*)


(* ::Input:: *)
(*mnistLikeDigits=ConformImages[Pick[images,#[[3]]=="flower"&/@annotations],{56,56},ColorSpace->"Grayscale"];*)


(* ::Input:: *)
(*Tally[ImageDimensions/@mnistLikeDigits]*)


(* ::Input:: *)
(*RandomSample[mnistLikeDigits,12]*)


(* ::Input:: *)
(*Sort[mnistLikeDigits]*)


(* ::Subsection:: *)
(*GAN*)


(* ::Input:: *)
(*ClearAll[progressFuncCreator]*)
(*progressFuncCreator[rands_List]:=Function[{reals},ImageResize[NetDecoder[{"Image","Grayscale"}][NetExtract[#Net,"gen"][reals]],50]]/@rands&*)


(* ::Input:: *)
(*trainingData=<|"random"->RandomInteger[{1,10},Length[mnistLikeDigits]],"Input"->Map[RandomReal[{-0.05,0.05},{1,28,28}]+ArrayReshape[ImageData[#],{1,28,28}]&,mnistLikeDigits]|>;*)
(*generator=NetChain[{EmbeddingLayer[8*6*6,10],ReshapeLayer[{8,6,6}],DeconvolutionLayer[8,4,"Stride"->2],Ramp,DeconvolutionLayer[1,4,"Stride"->2,"PaddingSize"->1],LogisticSigmoid}];*)
(*discriminator=NetChain[{ConvolutionLayer[4,4],Tanh,PoolingLayer[3,1],16,Ramp,1},"Input"->{1,28,28}];*)
(*wganNet=NetGraph[<|"gen"->generator,"discrimop"->NetMapOperator[discriminator],"cat"->CatenateLayer[],"reshape"->ReshapeLayer[{2,1,28,28}],"flat"->FlattenLayer[],"total"->SummationLayer[],"scale"->ConstantTimesLayer["Scaling"->{-1,1}]|>,{NetPort["random"]->"gen"->"cat",NetPort["Input"]->"cat","cat"->"reshape"->"discrimop"->"flat"->"scale"->"total"},"Input"->{1,28,28}];*)
(*NetTrain[wganNet,trainingData,"Output",Method->{"ADAM","Beta1"->0.5,"LearningRate"->0.01,"WeightClipping"->{{"discrimop",1}->1,"discrimop"->0.001}},TrainingProgressReporting->{progressFuncCreator[Range[10]],"Interval"->Quantity[0.3,"Seconds"]},LearningRateMultipliers->{"scale"->0,"gen"->-0.05},BatchSize->32,MaxTrainingRounds->5000];*)
