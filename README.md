# Project: Operationalizing Machine Learning Project on AWS

A program requirement for AWS Machine Learning Engineer Nanodegree @ Udacity School of Artificial Intelligence

## Project Description

This project invovles using several important tools and features of AWS to adjust, improve, configure, and prepare an image classification machine learning model for production-grade deployment on AWS. Taking raw ML code and preparing it for production deployment is a common task for ML engineers, and it's very important to ensure smooth, efficient, and optimal integration with other infrastructures and to ensure security.

## My Work

- Trained and deployed an image classification model on Amazon Sagemaker, using the most appropriate instances. 
- Set up multi-instance training in Sagemaker.
- Set up Sagemaker notebook to perform training and deployment to endpoint.
- Set up virtual server in cloud (EC2) inside a private VPC. Import data and train ML model on EC2
- Set up a Lambda function to invoke deployed model. 
- Set up auto-scaling for your deployed endpoint as well as concurrency for Lambda function to ensure latency and throughput.
- Ensured security of ML pipeline is set up properly by reviewing/assigning approriates IAM roles and policies.

## Sagemaker Instance

I used an ml.t3.medium instance for my Sagemaker instance. This, i believe, is sufficient computing resources to run my sagemaker notebook in reasonable time without incurring unbearable cost on AWS.

![Alt text](<Sagemaker Notebook instance-1.png>)

![Alt text](<Sagemaker studio-1.png>)

### Training Data and Output artifacts are uploaded to S3 bucket:
![Alt text](<S3 bucket-1.png>)

### Deployment

Models resulting from single-instance and multi-instance training are deployed to sagemeker endpoints:
![Alt text](<Sagemaker Endpoints-1.png>)


## Model Training on EC2

In addition to training my ML model on Sagemaker, I also trained a model with same data on AWS virtual server in cloud (EC2). I used the Amazon Machine Image  - Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1.0 (Ubuntu 20.04) 20231205. This is an OS container image with pre-installed pytorch package and other packages for machine learning and deep learning. Because cost is a major factor for me in executing this project, i initially opted for a free-tier Amazon EC2 t2.micro instance type. The drawback was the long time it took to train my ML model on the EC2 instance and the pipe broke before training could be completed. I then switch to an EC2 t3.medium instance type. Training was completed in minutes. Note the a Virtual private cloud was set up to is isolate my EC2 from the internet so that i could only use the EC2 with my local IP address.

![Alt text](<Model Training on EC2-1.png>)



The script used to train model on EC2 (ec2train1.py) is quite similar to the code (hpo.py) used as entry point in train_and_deploy-solution.ipynb. There are some differences in the modules used.  Although much of the EC2 training code has been adapted from hpo.py script, certain modules in hpo.py can only be used in SageMaker. Unlike ec2train.py, which trains a model with specific arguments, hpo.py parses arguments through the command line, enabling it to train multiple models with different hyperparameters.

## Lambda function for Endpoint Invocation
![Alt text](<Lambda function-1.png>)

Lamda function is created to take an input event (image url in dictionary format), process the input event and used that to invoke specified sagemaker endpoint currently in service in the AWS session. The lambda function is available in the lamdafunction.py script. The result of the invocation is further decoded. The result of the lambda function is returned to show statuscode and predictions of the endpoint. Each number represnts the log likelihood of the image belong to each class. The lambda function is test on AWS with the following response:

![Alt text](<Lambda function test response-1.png>)

Response
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "text/plain",
    "Access-Control-Allow-Origin": "*"
  },
  "type-result": "<class 'str'>",
  "Content-Type-In": "<__main__.LambdaContext object at 0x7fde2a714fa0>",
  "body": "[[-17.59284019470215, -9.236837387084961, -7.067824363708496, -4.12099027633667, -4.433526039123535, -14.484598159790039, -2.1540544033050537, -3.322157382965088, -12.38521957397461, -6.4254374504089355, -1.6744301319122314, -6.694097518920898, -5.838618278503418, -2.5299124717712402, -6.133054256439209, -4.222506523132324, -11.276371002197266, -3.184203863143921, -9.03726577758789, -3.316030740737915, -10.314987182617188, -5.746061325073242, -6.846641540527344, -9.6603364944458, -5.566239833831787, -11.959568977355957, -5.252140045166016, -4.744714260101318, -8.224257469177246, -3.331678628921509, -4.128255367279053, -6.471146106719971, -16.514236450195312, -6.342918872833252, -12.50295639038086, -14.9061861038208, -4.744229316711426, -7.763582706451416, -4.003729820251465, -9.714716911315918, -8.636046409606934, -10.69703483581543, -2.8951642513275146, -7.222990036010742, -3.753579616546631, -9.864822387695312, -6.309525489807129, -5.640625476837158, -5.954952239990234, -7.5314717292785645, -6.2659912109375, -9.554998397827148, -9.487929344177246, -6.337060928344727, -8.740092277526855, -2.4235095977783203, -8.197415351867676, -9.820588111877441, -3.2530739307403564, -7.93742036819458, -8.426139831542969, -7.673404216766357, -8.364325523376465, -9.917014122009277, -2.759237051010132, -13.528987884521484, -3.448087215423584, -8.416460990905762, -5.963663101196289, -4.425930023193359, -2.4709019660949707, -9.11512565612793, -8.42821979522705, -13.37205696105957, -10.515690803527832, -6.712928771972656, -9.36296558380127, -4.844581127166748, -10.398797988891602, -4.186346054077148, -2.8527112007141113, -13.446145057678223, -2.748706817626953, -5.597010612487793, -7.841383934020996, -9.355940818786621, -7.453980445861816, -9.9771089553833, -10.018013000488281, -4.597062110900879, -11.632560729980469, -15.885757446289062, -8.702770233154297, -14.304637908935547, -8.501331329345703, -5.489809036254883, -8.391162872314453, -8.87529182434082, -9.638789176940918, -11.9279203414917, -13.816865921020508, -4.794556617736816, -8.37659740447998, -8.259385108947754, -11.568593978881836, -11.298644065856934, -6.602653980255127, -4.13091516494751, -6.654895782470703, -4.33683967590332, -5.436748027801514, -5.037302494049072, -13.063488006591797, -10.894684791564941, -13.723236083984375, -4.799492835998535, -10.880433082580566, -4.0635666847229, -8.79650592803955, -4.124069690704346, -5.542270660400391, -3.8566808700561523, -12.335667610168457, -6.350172519683838, -14.653563499450684, -7.828548431396484, -7.16109561920166, -2.901899576187134, -6.048239707946777, -5.66408109664917, -13.537166595458984, -4.438143730163574, -10.689398765563965]]"
}

## Security and Integrity

The security and intergrity of my AWS project workspace is ensured by deploying to a private VPC (with configure security group and NACL) and only attaching essential permissions/policies to IAM roles, in every case "FullAccess" permission is thoughtfully considered before attaching. My IAM dashboard was thoroughly examined to clear out obsolete roles.

![Alt text](<IAM Dashboard-1.png>)

![Alt text](<Lambda function role policies-1.png>)

## Concurrency and Auto-scaling

Reserved concurrency for my lambda function was set to 10 invocations. I judged this to be sufficient for a personal project. Bearing cost in mind, just one of the 10 is set as provisioned concurrency. 
![Alt text](<Lambda concurrency setting-1.png>)

I chose a ml.m5.large instance and 2 maximum instance count for autoscaling the endpoint bearing in mind the need to balance having adequate computing power and cost. The target number of endpoint invocations to trigger autoscaling is set to 20 with 20 seconds of scale-in and scale-out cool down time for latency.
![Alt text](<Endpoint Autoscaling setting1-1.png>) ![Alt text](<Endpoint Autoscaling setting2-1.png>)

