/tool_lf/spark/spark-2.2.0-bin-hadoop2.7/bin/spark-submit\
 --py-files sample_model.py,AR_model_mapfunc.py,KDE_model_mapfunc.py\
 --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"\ 
 --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" \
 --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4"\ 
 model_run_AR.py\

#if [ $? -eq 'AR all over' ];then
#/tool_lf/spark/spark-2.2.0-bin-hadoop2.7/bin/spark-submit\
#--py-files sample_model.py,ekf_model_mapfunc.py,KDE_model_mapfunc.py \
#--conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64" \
#--conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" \
#--conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4" model_run_ekf.py\
#fi




