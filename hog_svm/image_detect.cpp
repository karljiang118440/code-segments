


HOGDescriptor hog()

MySVM svm;
svm.load();


DescriptorDim=svm.get_var_count();

int supportVectorNum=svm.get_support_vector_count();

Mat alphMat = Mat::zeros(1,supportVectorNum,CV_32FC1);
Mat supportVectorMat=Mat::zeros(supportVectorNum,DescriptorDim,CV_32FC1);
Mat resultMat=Mat::zeros(1,DescriptorDim,CV_32FC1);


for(int i=0;i<supportVectorNum;i++)
{
	const float* pSVData=svm.get_support_vector(i);
	for(int j=0;j<DescriptorDim;i++)
	{
		supportVectorMat.at<float>(i,j)=pSVData[j];
	}
}


// alpha 向量复制到 supportVectorMat 矩阵中
double *pAlphaData=svm.get_alpha_vector();
for(int i=0;i<supportVectorNum;i++)
{
	alphMat.at<float>(0,i)=pAlphaData[i];
}



// 计算resultMat，是由（-alphMat * supportVectorMat）得到
resultMat=-1*alphMat*supportVectorMat;

//得到 setSVMDetector(const vector<float>&detector)参数总可用的检测子
vector<float> myDetector;
//resultMat放到 myDetector
for(i=0;i<DescriptorDim;i++)
{
	myDetector.push_back(resultMat.at<float>(0,i));
}


//增加偏置量rho,得到检测子
myDetector.push_back(svm.get_rho());

//设置HOGDesriptor的检测子
HOGDescriptor myHOG;
myHOG.setSVMDetector(myDetector);

//保存检测子参数到文件




vector<Rect> found,found_filtered;
myHOG.detectMultiScale(img,found,0,Size(8,8),Size(8,8),1.05,2);

size_t i,j;
for(i=0;i<found.size();i++)
{
    Rect r=found[i];
    for(j=0;j<found.size();j++)
        if(j!=i&&(r&found[j])==r)
            break;
    if(j==found.size())
        found_filtered.push_back(r);
}
for(i=0;i<found_filtered.size();i++)
{
  Rect r=found_filtered[i];
  r.x+=cvRound(r.width*0.1);
  r.width=cvRound(r.width*0.8);
  r.y+=cvRound(r.height*0.07);
  r.height=cvRound(r.height*0.8);

  rectangle(img,r.tl(),r.br(),cv::Scalar(0,255,0),3);
}



}

