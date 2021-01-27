import configparser
from datetime import datetime
import os
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime as dt

def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    print("processing song data -->>>>")
    # get filepath to song data file
    song_data = input_data + "song_data/A/*/*/*.json"

    schema = StructType([
        StructField("artist_id",StringType()),
        StructField("artist_latitude",DoubleType()),
        StructField("artist_location",StringType()),
        StructField("artist_longitude",DoubleType()),
        StructField("artist_name",StringType()),
        StructField("duration",DoubleType()),
        StructField("num_songs",LongType()),
        StructField("song_id",StringType()),
        StructField("title",StringType()),
        StructField("year",LongType())
    ])
    
    # read song data file
    df = spark.read.json(song_data, schema=schema)

    df.createOrReplaceTempView("song_data")

    # extract columns to create songs table
    songs_table = spark.sql("""
        select 
        song_id, 
        title, 
        artist_id, 
        year, 
        duration,
        artist_name,
        year as partition_year 
        from 
        song_data""")
    
    # write songs table to parquet files partitioned by year and artist
    song_parquet_out = output_data + "songs.parquet"
    songs_table.write \
        .mode("append") \
        .partitionBy("partition_year", "artist_name") \
        .parquet(song_parquet_out)

    # extract columns to create artists table
    artists_table = spark.sql(""" select 
        artist_id, 
        artist_name as name, 
        artist_location as location, 
        artist_latitude as latitude, 
        artist_longitude as longitude 
        from
        song_data """)
    
    # write artists table to parquet files
    artist_parquet_out = output_data + "artists.parquet"
    artists_table.write \
        .mode("append") \
        .parquet(artist_parquet_out)


def process_log_data(spark, input_data, output_data):
    print("processing log data -->>>>")
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    log_schema = StructType([
        StructField("artist",StringType()),
        StructField("auth",StringType()),
        StructField("firstName",StringType()),
        StructField("gender",StringType()),
        StructField("itemInSession",LongType()),
        StructField("lastName",StringType()),
        StructField("length",DoubleType()),
        StructField("level",StringType()),
        StructField("location",StringType()),
        StructField("method",StringType()),
        StructField("page",StringType()),
        StructField("registration",StringType()),
        StructField("sessionId",StringType()),
        StructField("song",StringType()),
        StructField("status",IntegerType()),
        StructField("ts",LongType()),
        StructField("userAgent",StringType()),
        StructField("userId",StringType())
    ])

    # read log data file
    df = spark.read.json(log_data, schema = log_schema, multiLine="true")
    
    # filter by actions for song plays
    df.createOrReplaceTempView("log_data")

    # extract columns for users table    
    users_table = spark.sql("""select 
        CAST(userId as INT) as user_id, 
        firstName as first_name, 
        lastName as last_name, 
        gender,
        level 
        from
        log_data
        where page ='NextSong'""")
    
    # write users table to parquet files
    user_parquet_out = output_data + "users.parquet"
    users_table.write \
        .mode("append") \
        .parquet(user_parquet_out)
    
    # extract columns to create time table
    time_table = spark.sql("""select 
        ts as ts,
        from_unixtime(ts/1000,'yyyy-MM-dd HH:mm:ss') as start_time,
        from_unixtime(ts/1000,'H') as hour,
        from_unixtime(ts/1000,'dd') as day,
        from_unixtime(ts/1000,'MMM') as month,
        from_unixtime(ts/1000,'y') as year,
        from_unixtime(ts/1000,'E') as weekday,
        from_unixtime(ts/1000,'y') as partition_year,
        from_unixtime(ts/1000,'MMM') as partition_month
        from
        log_data
        where page ='NextSong'""")

    # create timestamp column from original timestamp column
    convertToWeekDay = F.udf(lambda z: dt.fromtimestamp(z/1000).isocalendar()[1], IntegerType())
    time_table = time_table.withColumn('week', convertToWeekDay(F.col('ts')))
    time_table = time_table.drop('ts')
    
    # write time table to parquet files partitioned by year and month
    time_parquet_out = output_data + "time.parquet"
    time_table.write \
        .mode("append") \
        .partitionBy("partition_year","partition_month") \
        .parquet(time_parquet_out)

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql(""" select
        from_unixtime(ts/1000,'yyyy-MM-dd HH:mm:ss') as start_time,
        CAST(userId as INT) as user_id,
        level,
        s.song_id as song_id,
        s.artist_id as artist_id,
        sessionId as session_id,
        location as location,
        userAgent as user_agent,
        from_unixtime(ts/1000,'MMM') as month,
        from_unixtime(ts/1000,'y') as year
        from log_data l
        left join 
        song_data s
        on l.artist = s.artist_name 
        where page ='NextSong'
    """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table = songplays_table.withColumn("songplay_id",F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))

    songplays_parquet_out = output_data + "songplays.parquet"
    songplays_table.write\
        .mode("append") \
        .partitionBy("year","month") \
        .parquet(songplays_parquet_out)

def main():
    config = configparser.ConfigParser()
    config.read('dl.cfg')

    os.environ["AWS_ACCESS_KEY_ID"]= config['AWS']['AWS_ACCESS_KEY_ID']
    os.environ["AWS_SECRET_ACCESS_KEY"]= config['AWS']['AWS_SECRET_ACCESS_KEY']
    
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://data-engineering-files/parquet/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
