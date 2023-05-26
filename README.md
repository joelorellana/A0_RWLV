# A0 Mode Classificators for Deep Dive Analysis
## _A further step of sentiment analysis._
###### Doc prepared by joelorellana989@gmail.com

## 1. Introduction.
This is the documentation of all models analyzed when I did all classifications tasks needed for hotel reviews.

As the available data from hotel reviews was not pre-classified, a technique called zero-shot classification needs to be employed. Zero-shot classification is a natural language processing task that enables the prediction of membership to a specific class without having pre-existing classifications.

This task presents certain challenges as it entails searching for suitable pre-trained models capable of performing the desired classifications. The process can be demanding, requiring thorough exploration and evaluation of various models to ensure their effectiveness in achieving the desired results.

The models utilized in this scenario are specific pre-trained models known for their successful performance. The most recent text classification model used is GPT-3.5 Turbo, which has demonstrated superior results compared to free models. Not only does it outperform other models in terms of accuracy, but it also boasts faster processing speed, making it highly efficient. Furthermore, GPT-3.5 Turbo proves to be cost-effective, minimizing the resources required for implementation.

In addition to GPT-3.5 Turbo, a neural network-based model called "pysentimiento" has been employed for various classification tasks. This model has been trained extensively on diverse datasets and exhibits comparable performance to GPT in terms of accuracy and reliability. By leveraging pysentimiento, reliable classification results can be obtained, contributing to the overall effectiveness of the classification process.

It is important to note that using pre-trained models like GPT-3.5 Turbo and pysentimiento eliminates the need for manual training from scratch. These models are already equipped with a wealth of knowledge and linguistic capabilities, making them powerful tools for classification tasks.

## 2. Models used.
### 2.1. For Sentiment Analysis

A simple yet effective classification approach has been adopted to categorize the reviews into positive (**POS**), negative (**NEG**), and neutral (**NEU**) sentiments. Considering the star ratings commonly used for hotels, the following mapping has been applied:

Reviews with ratings from 1 star ★ to 2 stars ★★ are classified as **NEG** 
Reviews with a 3-star ★★★ rating are classified as **NEU**
Reviews with ratings from 4 stars ★★★★ to 5 stars ★★★★★ are classified as **POS.**

This straightforward mapping allows for a clear and intuitive classification of the reviews based on the star ratings associated with hotels. It provides a practical framework for assessing the sentiment expressed in the reviews, aiding in the analysis and interpretation of customer experiences.

However, caution must be exercised as accurately identifying neutral reviews can be challenging, something that most models struggle with proper classification.

#### 2.1.1 Sentiment based on ratings
The Python function for classifying reviews is as follows:

```python
# Sentiment based on ratings
def get_sentiment_rating(rating):
  if rating == 0 or rating == 1:
    return 'NEG'
  elif rating == 4 or rating == 5:
    return 'POS'
  else:
    return 'NEU'
```
It's important to note that the identification of neutral reviews poses a difficulty for many models, making it a challenging task. The provided Python function, get_sentiment_rating(), allows for the classification of reviews based on their ratings. If the rating is 0 or 1, the sentiment is classified as NEG (negative). If the rating is 4 or 5, the sentiment is classified as POS (positive). For ratings other than these, the sentiment is classified as NEU (neutral).

This function provides a simplistic approach to assigning sentiment based on ratings. However, it's important to acknowledge that accurately identifying neutral reviews can be a complex task, as they often contain elements of both positive and negative sentiments.

#### 2.1.2 Sentiment Analysis with GPT3.5-Turbo.
##### 2.1.2.1 Introduction
The `get_sentiment_gpt` function is designed to classify the sentiment of hotel reviews using the GPT-3.5 Turbo model developed by OpenAI. This model is widely acclaimed for its advanced natural language processing capabilities, making it an effective choice for sentiment analysis tasks. By leveraging the GPT-3.5 Turbo model, the function aims to provide accurate sentiment classifications for hotel reviews.
##### 2.1.2.2 Installation and required libraries
To use the `get_sentiment_gpt` function, the openai library needs to be installed in the Python environment. This can be achieved by executing the following command:
```sh
pip install openai
```
Additionally, an API key from OpenAI is necessary for authentication and accessing the GPT-3.5 Turbo model. It is recommended to obtain the API key from OpenAI's website and configure it properly before utilizing the function.

Here's the import statement for the openai library and the api key required:
```python
import openai
openai.api_key = YOUR_KEY
```
And finally, here's the code for the sentiment analysis function:
```python
def get_sentiment_gpt(text):
    model_engine = 'gpt-3.5-turbo'
    prompt = 'Classify sentiment of the following hotel review. Mark the classification as positive, neutral or negative. Return only one word. Review: ' + text
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0,
    )
    output = response.choices[0]['message']['content'].lower()
    if 'positive' in output:
        return 'POS'
    elif 'negative' in output:
        return 'NEG'
    else:
        return 'NEU'
```
##### 2.1.2.3 Usage
The GPT-3.5 model can be used in various ways to perform sentiment analysis tasks. Here are a few examples of different input formats and options that can be used with the model:
###### Single Sentence Classification:
In this case, you provide a single sentence as input and the model classifies the sentiment as positive, neutral, or negative.
Here's an example:
```python
def get_sentiment_gpt(text):
    model_engine = 'gpt-3.5-turbo'
    prompt = 'Classify sentiment of the following sentence. Mark the classification as positive, neutral, or negative. Sentence: ' + text
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0,
    )
    output = response.choices[0]['message']['content'].lower()
    if 'positive' in output:
        return 'POS'
    elif 'negative' in output:
        return 'NEG'
    else:
        return 'NEU'

text = "This movie is fantastic!"
sentiment = get_sentiment_gpt(text)
print(sentiment)  # Output: 'POS'
```
###### Document-Level Classification:
For longer documents or a collection of sentences, it can be performed a sentiment analysis at a document level. Here's an example:
```python
def get_sentiment_gpt(text):
    model_engine = 'gpt-3.5-turbo'
    prompt = 'Classify sentiment of the following document. Mark the classification as positive, neutral, or negative. Document: ' + text
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0,
    )
    output = response.choices[0]['message']['content'].lower()
    if 'positive' in output:
        return 'POS'
    elif 'negative' in output:
        return 'NEG'
    else:
        return 'NEU'

document = "The service was great, but the food was disappointing. The hotel room was spacious and clean."
sentiment = get_sentiment_gpt(document)
print(sentiment)  # Output: 'NEU'
```
##### 2.1.2.4 Parameters

The function utilizes the GPT-3.5-turbo model to perform sentiment classification on a given hotel review text. It follows a chat-based API approach to interact with the model. The function accepts the following parameters:

**_text_** (string): The hotel review text that needs to be classified for sentiment analysis.

Returns:
**_sentiment_** (string): The sentiment classification of the hotel review, represented as a single word: 'POS' for positive, 'NEG' for negative, and 'NEU' for neutral.

**Model parameters**:

1. **_model_engine_** (string):
**Default value:** 'gpt-3.5-turbo'
**Description**: Specifies the GPT model variant to use. In this case, it uses the GPT-3.5-turbo model.

2. **_prompt_** (string):
**Description**: The prompt presented to the model. It includes instructions for sentiment classification and the provided hotel review text.

3. **_messages_** (list of dictionaries):
**Description**: Represents the conversation history or messages given to the model during the interaction.
**Structure of each message dictionary**:
**_'role'_** (string): Specifies the role of the message, which can be '_system_', '_user_', or '_assistant_'.
**_'content'_** (string): The content or text of the message.
4. **max_tokens** (integer):
**Default value**: 1
**Description**: Specifies the maximum number of tokens in the model's response. In this case, it limits the response to a single word representing the sentiment classification.
5. **_n_** (integer):
**Default value**: 1
**Description**: Specifies the number of responses to generate from the model.
6. **_stop_** (string or list of strings):
**Default value**: None
**Description**: Specifies the stopping criteria for the generated text. If the generated text contains any of the specified strings in the stop list, the response will be truncated at that point.
7. **_temperature_** (float):
Default value: 0
Description: Controls the randomness of the model's output. A higher value (e.g., 1.0) makes the output more random, while a lower value (e.g., 0.2) makes it more deterministic.

##### 2.1.2.5 Performance and Limitations:

##### Performance characteristics:

###### Speed:
The GPT-3.5-turbo model is designed to provide fast response times. In most cases, generating a response typically takes a few seconds. However, the exact response time can vary based on factors such as the length and complexity of the input text, the number of tokens requested, and the current load on the OpenAI API.

During the implementation of the different classifications for hotel reviews, several challenges arose, such as the Limit Rate Error due to the overload of queries to the GPT model. As a result, the approach of executing the classifications in batches of reviews was adopted. Due to these limitations, it is currently recommended to perform these classifications under human supervision rather than automating them automatically.

This batch execution approach allows processing a group of hotel reviews at once, which helps avoid model saturation and ensures more stable execution. However, it is important to note that this approach requires human intervention to supervise and control the classification process.

By performing hotel review classifications under human supervision, additional adjustments and validations can be made as necessary. This ensures higher accuracy and reliability in sentiment classification results.

It is crucial to consider that these recommendations are subject to changes and updates based on resource availability and the evolution of the OpenAI platform. It is advised to follow the guidelines and best practices provided by OpenAI to ensure effective and optimal use of the GPT-3.5-turbo model in hotel review classification.

###### Memory Requirements:
The GPT-3.5-turbo model requires memory to store the model parameters and process the input data. The memory requirements can vary depending on the input text length and the specific computational infrastructure used. It's important to ensure that your system has enough memory available to handle the model's requirements.

###### Scalability:
The GPT-3.5-turbo model is designed to be highly scalable. It can handle a large number of concurrent requests, allowing for efficient parallelization. OpenAI's infrastructure is optimized to handle high demand, but it's essential to consider the specific usage limits and rate limits defined by OpenAI when deploying the model at scale.

##### Known Limitations and Potential Pitfalls:

###### Lack of Contextual Understanding:
While GPT-3.5-turbo is a powerful language model, it may not have a deep understanding of the specific context or domain. It relies on statistical patterns learned from training data and might generate responses that sound plausible but lack true comprehension of the content. Care should be taken to validate and verify the generated sentiments in critical applications.

###### Sensitivity to Input Phrasing:
The model's responses can be sensitive to slight changes in input phrasing or prompt formulation. The choice of words, grammar, and sentence structure can influence the generated sentiment classification. Experimentation and careful refinement of prompts may be necessary to achieve desired results.

###### Biased Outputs:
Like other language models, GPT-3.5-turbo can inadvertently reflect biases present in the training data. It is important to be cautious and consider the potential biases in the generated sentiment classifications, especially when used in sensitive or high-stakes applications. Post-processing and bias mitigation techniques can be applied to address this concern.

###### Handling Extremely Long Texts:
The GPT-3.5-turbo model has a maximum token limit, and if the input text exceeds this limit (4096 tokens), it needs to be truncated or split into smaller parts. However, splitting long texts may affect the context and result in less accurate sentiment classifications, particularly when the sentiment-bearing information is spread across multiple segments.

##### 2.1.2.6 Troubleshooting:
In the event of a **_rate limit error_**, it is advisable to wait for a short period of time before attempting to execute again. It is also recommended to reduce the size of the batches being processed. Once the errors no longer occur, gradually increasing the batch size can be considered.

Waiting for a short duration allows for the rate limit to reset and helps prevent further rate limit errors. Reducing the batch size helps to decrease the number of API calls made within a given time frame, reducing the likelihood of hitting the rate limit.

After waiting and successfully executing with smaller batches, gradually increasing the batch size can be done as long as no rate limit errors are encountered. This approach allows for a controlled scaling of the workload while avoiding rate limit issues.

It is important to note that the specific duration to wait and the optimal batch size may vary depending on the rate limits set by the OpenAI API and the specific requirements of the implementation. Monitoring the API responses and adjusting accordingly will help ensure a smooth and efficient execution without exceeding the rate limits.

##### 2.1.2.7 Working in batches
It is highly recommended to operate using incremental batches due to potential saturation issues with OpenAI's API when classifying substantial volumes of text such as reviews, which tend to be lengthy.
For batch classification, first create a classification column with placeholder values, like `None`.
```python
df['sentiment'] = None
```
To apply batch classification incrementally to a column, it is suggested to start with 50 classifications and gradually double this number until you encounter any saturation. For example, the first time you execute the classification, using the function `get_sentiment_gpt()`, an update of the column can be performed with the `update` command, to prevent memory duplications in the dataframe.
```python
df['sentiment'].update(df.comments.iloc[START:END].apply(get_sentiment_gpt))
```
After that, gradually increment the number of rows in the `.iloc[START:END]`parameter. 


##### 2.1.2.8 References 
For further explorations and improvements about models including the access to the new and upcoming GPT 4 API, visit: [OpenAI documentation](https://platform.openai.com/overview)
Book used for testings and prompts as used in this project: [Exploring GPT-3 ](https://www.packtpub.com/product/exploring-gpt-3/9781800563193)
Technical paper about the upcoming GPT-4: [GPT-4 Tech Paper ](https://cdn.openai.com/papers/gpt-4.pdf)

#### 2.1.3 Other models tested for sentiment analysis
Below are other models considered for sentiment analysis but were discarded in the final presentation because they performed worse than the previously mentioned GPT model. However, these are traditional sentiment analysis models that can be considered if GPT is not used. It is recommended to perform confusion matrix tests and classification reports before using any model.
#### 2.1.3.1 Pysentimiento model
After GPT, this is the model with better results than others. Made in Pytorch is a Deep Learning model for several NLP tasks.
Here is the complete code for implementation:
First, library installation:
```sh
pip install pysentimiento
```
And the model as function is:

```python
from pysentimiento import create_analyzer
sentiment = create_analyzer(task="sentiment", lang="en")
def get_sentiment_pys(text):
    return sentiment.predict(text).output
```
For a detailed explanation of pysentimiento see the next section.

#### 2.1.3.2 Textblob model
The next model utilizes the TextBlob library, a widely-used Python library for natural language processing tasks, including sentiment analysis. The purpose of this model is to determine the sentiment (positive, neutral, or negative) of a given text input.

The next code is the implementation of the model:
First, library installation:
```sh
pip install textblob
```
An here is the function for the textblob model:

```python
def get_sentiment_label_textblob(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'POS'
    elif sentiment == 0:
        return 'NEU'
    else:
        return 'NEG'
```

The function `get_sentiment_label_textblob` takes a parameter called `text`, which represents the input text to be analyzed. The sentiment analysis is performed by utilizing TextBlob's built-in functionality. Specifically, the `sentiment.polarity` attribute of TextBlob is used to calculate the polarity of the text.

The sentiment polarity value returned by TextBlob ranges between -1 and 1. A polarity greater than 0 indicates a positive sentiment, resulting in the function returning the label 'POS'. A polarity equal to 0 signifies a neutral sentiment, leading to the function returning the label 'NEU'. Lastly, a polarity less than 0 suggests a negative sentiment, and the function returns the label 'NEG'.

It is important to note that TextBlob's sentiment analysis relies on a pre-trained model and lexical resources, such as WordNet, to evaluate sentiment based on word frequencies and associations. However, this model lacks context and a deep understanding of the semantic meaning of the text, which more advanced models may incorporate.

While TextBlob's sentiment analysis can be useful for simple sentiment classification tasks, it may not be as accurate or robust as more sophisticated models, especially when dealing with complex or domain-specific language. The performance and accuracy of this model heavily depend on the quality and relevance of the underlying pre-trained model and lexical resources utilized by TextBlob.

#### 2.1.3.3 Vader Sentiment Analysis 
First, we need to install vader in Python:
```sh
pip install vaderSentiment
```
And the function that contains the Vader model is:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()
def get_sentiment_vader(text):
    score = vader_analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return 'POS'
    elif -0.05 < score < 0.05:
        return 'NEU'
    else:
        return 'NEG'
```

The previous code  implements a sentiment analysis model using the VaderSentiment library. VaderSentiment is a popular Python library specifically designed for sentiment analysis, particularly suited for social media texts.

The `get_sentiment_vader` function takes a `text` parameter, which represents the input text to be analyzed for sentiment. The sentiment analysis is performed using the `SentimentIntensityAnalyzer` class from the VaderSentiment library.

First, the sentiment intensity analyzer (`vader_analyzer`) calculates the sentiment score for the input text using the `polarity_scores()` method. The `compound` score is extracted from the returned dictionary, which represents the overall sentiment polarity ranging from -1 to 1. The compound score takes into account the aggregated sentiment across the text.

Next, the function applies a set of conditions to determine the sentiment label. If the sentiment score is greater than or equal to 0.05, it is considered positive, and the function returns the label 'POS'. If the score falls between -0.05 and 0.05 (exclusive), it is considered neutral, and the function returns the label 'NEU'. For scores lower than -0.05, the sentiment is considered negative, and the function returns the label 'NEG'.

VaderSentiment utilizes a pre-trained model based on a combination of lexical features and heuristics. It incorporates sentiment lexicons and grammatical rules to assess sentiment intensity. It is designed to handle sentiment analysis for informal, short texts typically found in social media posts.

It is important to note that while VaderSentiment performs well in many cases, it may not capture certain nuances or understand context as accurately as more advanced models. Its performance is highly dependent on the quality and relevance of the underlying sentiment lexicons and rules.

Therefore, before utilizing this sentiment analysis model, it is recommended to evaluate its performance on the specific dataset or domain of interest. Assessing metrics such as accuracy, precision, recall, and F1-score, as well as considering potential limitations or edge cases, can provide a more comprehensive understanding of its suitability for the task at hand.

### 2.2 For emotion analysis
#### Pysentimiento model for emotion.
#### 2.2.1 Introduction
pysentimiento is a pre-trained neural network library to implement several natural language processing tasks including an emotion analysis model. This model is designed to analyze and classify the emotions expressed in a given text. The goal is to identify the underlying emotions conveyed by the text, allowing for a deeper understanding of the emotional context.

#### 2.2.2 Installation and system requirements

pysentimiento was made based on PyTorch, a deep learning framework in Python.
Due to its nature, pysentimiento requires downloading all the models to be used, depending on the type of classification desired. pysentimiento requires a minimum of 4GB of available storage and at least 6GB of available RAM. Due to its resource consumption, it is recommended to use pysentimiento on cloud virtual machines with an activated GPU.
To install pysentimiento in console run:
```sh
pip install pysentimiento
```
After installing pysentimiento, proceed with importing the analyzer and create a function for the model:
```python
from pysentimiento import create_analyzer
emotion = create_analyzer(task="emotion", lang="en")
def get_emotion_pys(text):
    return emotion.predict(text).output
```
#### 2.2.3 Usage

Pysentimiento can be used for several NLP tasks in several languages, the next table taken from GitHub docs detailed all its common tasks.
Currently supports:


| Task                                 | Languages                             |
|:---------------------                |:---------------------------------------|
| Sentiment Analysis                   | es, en, it, pt                        |
| Hate Speech Detection                | es, en, it, pt                        |
| Irony Detection                      | es, en, it, pt                        |
| Emotion Analysis                     | es, en, it                            |
| NER & POS tagging                    | es, en                                |
| Contextualized Hate Speech Detection | es                                    |
| Targeted Sentiment Analysis          | es                                    |

Here are some examples of use:

```python
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")

analyzer.predict("Qué gran jugador es Messi")
# returns AnalyzerOutput(output=POS, probas={POS: 0.998, NEG: 0.002, NEU: 0.000})
analyzer.predict("Esto es pésimo")
# returns AnalyzerOutput(output=NEG, probas={NEG: 0.999, POS: 0.001, NEU: 0.000})
analyzer.predict("Qué es esto?")
# returns AnalyzerOutput(output=NEU, probas={NEU: 0.993, NEG: 0.005, POS: 0.002})

analyzer.predict("jejeje no te creo mucho")
# AnalyzerOutput(output=NEG, probas={NEG: 0.587, NEU: 0.408, POS: 0.005})
"""
Emotion Analysis in English
"""

emotion_analyzer = create_analyzer(task="emotion", lang="en")

emotion_analyzer.predict("yayyy")
# returns AnalyzerOutput(output=joy, probas={joy: 0.723, others: 0.198, surprise: 0.038, disgust: 0.011, sadness: 0.011, fear: 0.010, anger: 0.009})
emotion_analyzer.predict("fuck off")
# returns AnalyzerOutput(output=anger, probas={anger: 0.798, surprise: 0.055, fear: 0.040, disgust: 0.036, joy: 0.028, others: 0.023, sadness: 0.019})

"""
Hate Speech (misogyny & racism)
"""
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")

hate_speech_analyzer.predict("Esto es una mierda pero no es odio")
# returns AnalyzerOutput(output=[], probas={hateful: 0.022, targeted: 0.009, aggressive: 0.018})
hate_speech_analyzer.predict("Esto es odio porque los inmigrantes deben ser aniquilados")
# returns AnalyzerOutput(output=['hateful'], probas={hateful: 0.835, targeted: 0.008, aggressive: 0.476})

hate_speech_analyzer.predict("Vaya guarra barata y de poca monta es XXXX!")
# returns AnalyzerOutput(output=['hateful', 'targeted', 'aggressive'], probas={hateful: 0.987, targeted: 0.978, aggressive: 0.969})
```



##### Preprocessing

`pysentimiento` features a tweet preprocessor specially suited for tweet classification with transformer-based models.

```python
from pysentimiento.preprocessing import preprocess_tweet

# Replaces user handles and URLs by special tokens
preprocess_tweet("@perezjotaeme debería cambiar esto http://bit.ly/sarasa") # "@usuario debería cambiar esto url"

# Shortens repeated characters
preprocess_tweet("no entiendo naaaaaaaadaaaaaaaa", shorten=2) # "no entiendo naadaa"

# Normalizes laughters
preprocess_tweet("jajajajaajjajaajajaja no lo puedo creer ajajaj") # "jaja no lo puedo creer jaja"

# Handles hashtags
preprocess_tweet("esto es #UnaGenialidad")
# "esto es una genialidad"

# Handles emojis
preprocess_tweet("🎉🎉", lang="en")
# 'emoji party popper emoji emoji party popper emoji'
```
#### 2.2.4 Parameters
pysentimiento doesn't have aditional parameters to configure because it's a trained model. Developers can get access to dataset for train aditional parameters but need permissions which are given via email requests.

#### 2.2.5 Performance and limitations
There are no known bugs showed in all executions of pysentimiento in this project. GitHub project says pysentimiento needs python <= 3.10 to work properly.
For system requirements of pysentimiento see section 2.2.2

#### 2.2.6 References
More info about pysentimiento and future tasks can be searched at:
GitHub pysentimiento project: [GitHub page ](https://github.com/pysentimiento/pysentimiento)
Paper of pysentimiento: [pysentimiento paper ](https://arxiv.org/abs/2106.09462)
Page at huggingfaces: [HugginFaces pysentimiento](https://huggingface.co/pysentimiento)

### 2.3 For contextual analysis
Before we can train a Machine Learning model for future classifications, we need data labeled accurately so a multi labeling classificator can be implemented.
Fortunately we can classify data with the same model of GPT used for sentiment analysis with some changes to the prompt reflecting what we're tryng to do.

#### 2.3.1 Code for contextual analysis
Next is the `get_contextual` function which contains the prompt and the GPT model used for classification.

```python
def get_contextual(text):
    model_engine = 'gpt-3.5-turbo'
    prompt = "From the next categories: Location, Acommodation, Staff, Food, Facilities, Price, Noise, Safety, Accesibility, Pool, Concert.\n Classify the following review in only one, the most related category and return it in only one word, if there is another category not mentioned, return it in one word:\n" + text
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0,
    )
    contextual = response.choices[0]['message']['content'].lower().replace('.', '')
    if contextual in ['location', 'acommodation', 'staff', 'food', 'facilities', 'price', 'noise', 'safety',
                      'accesibility', 'pool', 'concert']:
        return contextual
    else:
        return contextual.split(' ')[0]
```

#### 2.3.2 Variables and parameters

Since the model's parameters have been explained in the section dedicated to sentiment analysis, the following provides an explanation of all the variables and parameters involved. For a more detailed explanation, please refer to the previously mentioned section.

1. The function takes a parameter `text`, which represents the input review text to be classified.

2. It defines the `model_engine` variable, which specifies the GPT-3.5-turbo model to be used.

3. The `prompt` variable is constructed by concatenating the provided `text` with a predefined prompt message. The prompt message asks the user to classify the review into one of the given categories (Location, Accommodation, Staff, Food, Facilities, Price, Noise, Safety, Accessibility, Pool, Concert) and return it in one word. If the review relates to a category not mentioned, the user should return it in one word as well.

4. The `openai.ChatCompletion.create` method is called to generate a response using the GPT-3.5-turbo model. It takes the model, messages (containing the user prompt), and additional configuration parameters.

5. The `max_tokens` parameter limits the length of the generated response to 10 tokens to ensure concise output.

6. The `n` parameter specifies that only one message response is requested from the model.

7. The `stop` parameter is set to `None`, allowing the model to ge`n`erate a complete response without any predefined stopping condition.

8. The `temperature` parameter is set to 0, indicating a deterministic output without randomness.

9. The generated response is stored in the `response` variable.

10. The response is processed to extract the generated classification, stored in the `contextual` variable. It is converted to lowercase and any trailing periods are removed.

11. The `contextual` value is checked against the predefined category list. If it matches any of the categories, that category is returned.

12. If the `contextual` value does not match any of the categories, it is assumed to contain multiple words. In this case, the first word is extracted and returned as the category.

### 2.4 For intention analysis
Intent analysis refers to the process of determining the underlying intention or purpose behind a given text or statement. It involves categorizing the text into different predefined categories based on the intended meaning. The purpose of intent analysis is to extract valuable insights from textual data and understand the user's intention or desired action.

#### 2.4.1 Code for intention analysis
The next `get_intention()` function aims to perform intent analysis on a given text. It utilizes the GPT-3.5-turbo model to classify the input text into one of several predefined categories. The function returns the most relevant category as a single word, and if the input text belongs to a category not explicitly mentioned, it provides a general representation of that category in a single word.

```python
def get_intention(text):
    model_engine = "gpt-3.5-turbo"
    prompt = "From the next categories: complaint, opinion, compliment, feedback, suggestion, spam or random.\n Classify the following review in only one, the most related category and return it in only one word, if there is another category not mentioned, return it in one word:\n" + text
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0,
    )
    intention = response.choices[0]['message']['content'].lower().replace('.', '')
    if intention in ['complaint', 'compliment', 'opinion', 'suggestion', 'spam', 'random', 'feedback']:
        return intention
    else:
        return intention.split(' ')[0]
```

#### 2.4.2 Model parameters and overview
Given the previous discussions on the configuration parameters of GPT in earlier sections, the following will solely focus on explaining the specific details that differentiate this function from others.

The `get_intention()` function takes a parameter called text, which represents the input text to be analyzed for intent. It combines the input text with a prompt message that instructs the user to classify the review into one of the provided categories. The goal is to identify the most related category and return it as a single word. If the input text belongs to a category not explicitly mentioned, the function still returns a representation of that category in a single word.

**Model Invocation**:
To perform the intent analysis, the function utilizes the GPT-3.5-turbo model, which is specified as `the model_engine`. The `openai.ChatCompletion.create()` method is called to generate a response using the model. It passes the model, messages (containing the user prompt and input text), and additional configuration parameters such as `max_tokens`, `n`, `stop`, and `temperature`.

**Output Processing**
The `response` from the model is stored in the response variable. The function then extracts the content of the message from the response, converts it to lowercase, and removes any trailing periods. This processed content is stored in the `intention` variable.

**Category Validation**:
The `intention` value is checked against a list of predefined categories: complaint, compliment, opinion, suggestion, spam, random, and feedback. If the `intention` matches any of these categories, it is returned as the predicted intention.

If the `intention` value does not match any of the predefined categories, it is assumed to contain multiple words. In this case, the function extracts the first word and returns it as a representation of the category.

## 3. Summarization models
### 3.1 Sumy summarization models
Sumy is a simple library and command line utility for extracting summary from HTML pages or plain texts. The package also contains simple evaluation framework for text summaries.
#### 3.1.1 Introduction of summary models
TextRank and LSA Text Summarizer are models designed to provide concise summaries of English text. These models are implemented in the sumy library's implementation of the TextRank and Latent Semantic Analysis (LSA) summarization algorithms. They solve the problem of extracting the most important sentences from larger bodies of text.
#### 3.1.2 Installation
The sumy library is required for these models. It can be installes it using pip:
```sh
pip install sumy
```
After, it can be imported in python with this lines:
```python
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
```
These classes are detailed next:

1. `TextRankSummarizer`: This class is part of Sumy's summarization implementations. It applies the TextRank algorithm, which is a graph-based ranking algorithm for natural language processing and information retrieval. The algorithm identifies significant keywords in the text and assigns them a rank based on their interconnectedness. The `TextRankSummarizer` extracts the most relevant sentences from a document using these ranked keywords. It is typically utilized when a brief, coherent summary of a large document is required, and is beneficial in diverse areas like summarizing emails, articles, or web content.

2. `LsaSummarizer`: The `LsaSummarizer` class implements another summarization technique known as Latent Semantic Analysis (LSA). LSA is a technique in natural language processing that extracts and represents the contextual-usage meaning of words by statistical computations applied to a large corpus of text. It takes into account the syntactic relationships within the text and calculates the "importance" of sentences based on their semantic content. This summarizer may be used when more complex text understanding is required, including capturing implied meanings and relationships in the text that are not explicitly stated.

3. `PlaintextParser`: This class is responsible for parsing plain text data. The `PlaintextParser` takes in a string of plain text and parses it into a document that can be processed by Sumy's summarizers. The result is a structured object that represents the document, broken down into individual sentences and words. This class plays a crucial role in pre-processing, transforming raw text into a form suitable for summarization.

4. `Tokenizer`: The `Tokenizer` class is a part of Sumy's natural language processing utilities. It is used to break down the text into its constituent parts, called tokens, which can be as small as individual words. Tokenizing is a fundamental step in text processing, as it transforms unstructured data into a structured form that can be analyzed and manipulated. The `Tokenizer` is essential in preparing text for further processing, such as parsing and summarization.

These classes all play their unique roles within the Sumy library, working together to process and understand text data, and to generate high-quality summaries of that data.

#### 3.1.3 Usage 

In the following, the two functions incorporating the aforementioned summarization models are presented. These functions utilize distinct algorithms for the purpose of summarizing texts, providing two different approaches based on the specific requirements and complexity of the textual content.

The purpose of these functions is to reduce the length of the content while retaining the main points, crucial details, and overall essence of the original text. Each summarization model has its unique methodology and advantage, offering versatility in handling various types of text.

The functions operate by processing the input text, which involves tokenizing the text into individual words or phrases and parsing it into a structured format suitable for analysis. After this pre-processing stage, the summarization models come into play, analyzing the parsed text to determine the key points and thereby generate the summary.

These functions can be used independently based on the desired summarization technique - the TextRank algorithm or the Latent Semantic Analysis. The TextRank function is suited to extract a concise summary focusing on keywords, while the LSA function provides a more sophisticated understanding of the text, capturing implicit meanings and relationships.

However, it should be noted that the choice of function depends largely on the nature of the text and the desired outcome. For instance, if the goal is to quickly grasp the gist of a document, the TextRank function may be preferred. Conversely, if the aim is to derive a deeper understanding of the text, particularly in more complex or nuanced documents, the LSA function would likely provide better results.

**TextRank summarizer function**:
```python
def textrank_summarizer(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count=1)
    return str(summary[0])
```
**LSA summarizer function**:
```python
def lsa_summarizer(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=1)
    return str(summary[0])
```

The `textrank_summarizer` and `lsa_summarizer` functions are designed to accept a single parameter, a string of English text, and output its summary as a string.

Usage example:
```python
text = "This is a lengthy text to be summarized."
textrank_summary = textrank_summarizer(text)
lsa_summary = lsa_summarizer(text)
print("TextRank Summary:", textrank_summary)
print("LSA Summary:", lsa_summary)
```
The output would be the most informative sentence from the text according to the respective algorithms.

#### 3.1.4 Parameters
Both functions have the following parameter:

* **text**: The English text to be summarized. It must be a string.

The functions are currently set to output a one-sentence summary. It can be customized this by modifying the sentences_count parameter in the function definition.

#### 3.1.5 Performance and limitations
These functions are generally efficient and fast, thanks to the performance optimizations in the Sumy library. However, processing time can increase for larger inputs.

These functions are designed to work with English text. They may not provide optimal results with other languages. Furthermore, the current configuration returns only one sentence summary, which might not be sufficient for all types of desired tasks.

#### 3.1.6 References
Here are some additional resources that can be used as references when documenting the Sumy library:
* Official Sumy Documentation: [sumy](https://pypi.org/project/sumy/)
* Source Code on GitHub: [Source Code](https://github.com/miso-belica/sumy)
* TextRank Algorithm: [An explanation paper about TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
* Latent Semantic Analysis (LSA): [LSA tutorial using Python](https://www.datacamp.com/tutorial/discovering-hidden-topics-python)

### 3.2 Summarizer with Transformers
#### 3.2.1 BART summarizer
The Bidirectional and Auto-Regressive Transformers (BART) Text Summarizer is a function leveraging the capabilities of the BART model for text summarization tasks. The model serves the purpose of condensing extensive English text into a shorter, summarized form, facilitating a more efficient digestion of information.

The function has been developed based on the 'facebook/bart-large-cnn' variant of the BART model, which is renowned for its high performance in text summarization. It necessitates the installation of the Hugging Face Transformers library, which encapsulates the model and tokenizer used in the function.

#### 3.2.2 Installation
Installation of the Hugging Face Transformers library is essential for the function's operation. Python 3.6 or later is also required. The Transformers library can be installed through pip:
```python
pip install transformers
```
After successful installation of the Transformers library, the BART model and its associated tokenizer can be downloaded from the Hugging Face model hub. It is recommended to have a robust internet connection during this process, as the model files can be quite large.
The next code is the implementation of the BART summarizer model:
```python
from transformers import BartTokenizer, BartForConditionalGeneration
transformers_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
```

The first line imports the necessary classes from the Hugging Face Transformers library. `BartTokenizer` and `BartForConditionalGeneration` are specific components of the BART (Bidirectional and Auto-Regressive Transformers) model architecture. `BartTokenizer` is used to convert input text into a format (tokens) that the BART model can understand and process. `BartForConditionalGeneration` is a variant of the BART model designed for tasks that involve generating text based on certain conditions, such as text summarization.

#### 3.2.3 Usage
The function for summarization with BART summarizer is:
```python
def generate_long_summary(text):
    inputs = transformers_tokenizer.batch_encode_plus([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=40, min_length=10, early_stopping=True)
    summary = transformers_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```
The `generate_long_summary` function is designed to receive a string of English text as input, and produce a succinct summary of that text. The function begins by using the BART tokenizer to convert the text into a format compatible with the BART model. It then passes the tokenized input to the model, which generates a summary in tokenized form. Finally, the function decodes the tokenized summary back into human-readable text and returns it.

An example of how to employ the function is as follows:
```python
text = "This is an extensive text that needs summarization."
summary = generate_long_summary(text)
print(summary)
```
This would output the summarized version of the text, a concise interpretation of the original input.

#### 3.2.4 Parameters

The `generate_long_summary` function has one input parameter:

* **`text`**: The extensive English text that requires summarization. This must be in string format.

In the course of the function's operation, several other parameters are used. These are:
* **`max_length`** for both the tokenizer and the model, dictating the maximum length of the tokenized input text and the generated summary respectively. The default values are 1024 for the tokenizer and 40 for the model.
* **`min_length`** for the model, determining the minimum length of the generated summary. The default value is 10.
* **`num_beams`** for the model, controlling the number of beams for beam search in the model. The default value is 4.
* **`early_stopping`**, a Boolean that decides whether to stop the beam search when num_beams sentences are generated. The default value is True.

Each of these parameters can be adjusted in accordance with specific needs, but caution should be exercised as changes can affect the model's performance and the quality of the summaries.

#### 3.2.5 Performance and Limitations

The performance of the BART Text Summarizer is heavily contingent on the complexity and length of the input text, as well as the computational resources available. The BART model used ('facebook/bart-large-cnn') has been pretrained on a vast corpus of data and generally produces high-quality summaries. However, it is important to note that the model's performance can be influenced by the text input.

In terms of limitations, the model can handle a maximum input length of 1024 tokens due to the internal `max_length` parameter for the tokenizer. For very long texts exceeding this limit, it is necessary to either truncate the text or split it into smaller chunks before processing. Moreover, the generated summary has a default maximum length of 40 tokens and a minimum of 10 tokens. Depending on the use case, these limitations can affect the utility of the summarizer.

The model might occasionally produce summaries that miss important details from the original text, especially when dealing with texts that are complex or contain niche, domain-specific vocabulary. It's also worth noting that, like many NLP models, BART might not maintain the context over very long pieces of text.

#### 3.2.6 Troubleshooting
Users may encounter issues with memory usage and model loading times. This is common when working with large transformer models like BART. If memory-related errors occur, consider reducing the batch size or shortening the text input.

In case of problems during installation or model loading, verify the Python version (3.6 or later required) and the installation status of the Transformers library. Ensure a stable internet connection is available when downloading the model and tokenizer.

If the summaries generated are not satisfactory or lack important details, consider adjusting the internal parameters like `max_length`, `min_length`, or `num_beams` within the function. It's crucial to understand the impact of these parameters on the output and exercise due diligence while tweaking them.

#### 3.2.7 References

BART: [Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer.](https://arxiv.org/abs/1910.13461)
Hugging Face Transformers GitHub repository: [Detailed documentation and examples can be found in the official GitHub repository for the Transformers library.](https://github.com/huggingface/transformers)
Hugging Face Model Hub: [A hub of pretrained models in different languages and for various tasks, including the 'facebook/bart-large-cnn' model.](https://huggingface.co/models)

### 3.3 Summarizer with GPT
#### 3.3.1 Model GPT function for summarizer
The same GPT model used for sentiment analysis and other classifications will also be utilized for the task of text summarization. The respective function is presented as follows:
```python
def summarizer_gpt(text):
    model_engine = 'gpt-3.5-turbo'
    prompt = 'Given this hotel user review text, summarize the most important in approximate 10 words. Only use a point at the end of the summarization. ' + text
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=15,
        n=1,
        stop=['.'],
        temperature=0.7,
    )
    output = response.choices[0]['message']['content'].lower().rstrip('. ,')
    return output
```
#### 3.3.2 Model parameters

The `summarizer_gpt` function uses the GPT-3.5-turbo model from OpenAI for text summarization tasks. Here is a detailed breakdown of the function and its parameters:

* **`text`**: This is the only parameter that the summarizer_gpt function takes as input. It should be a string containing the text to be summarized.

* **`model_engine`**: This variable holds the identifier of the model to be used for text generation. The identifier 'gpt-3.5-turbo' corresponds to OpenAI's GPT-3.5-turbo model. Other options include 'text-davinci-003' or 'text-curie-003', but these will have different performance and cost characteristics.

* **`prompt`**: The `prompt` variable is a string that instructs the model to perform a specific task. It concatenates a command with the input text to guide the model in generating a summary.

* **`response`**: This line of code makes a request to the OpenAI API to generate a text based on the input `messages`, `model`, and additional parameters.

* **`model`**: The identifier of the model to be used, which in this case is the value stored in the model_engine variable.

* **`messages`**: A list of message objects. Each message object has a 'role' that can be 'system', 'user', or 'assistant', and 'content' which is the actual content of the message. In this case, there is one 'user' message containing the prompt.

* **`max_tokens`**: This parameter sets a limit to the length of the generated output. It is set to 15 in this instance, which means the model will generate a response that is no longer than 15 tokens.
* **`n`**: This parameter specifies the number of independent completions to generate from the prompt. A value of 1 means only a single completion is generated.

* **`stop`**: A list of sequences where the model will stop generating further tokens. In this case, it is set to ['.'], so the model will stop generating tokens once it outputs a period.

* **`temperature`**: This parameter controls the randomness of the model's output. A higher temperature results in more random outputs, while a lower temperature makes the output more focused and deterministic. The temperature is set to 0.7 in this case, leaning towards deterministic but with some level of variability.

The `response` object is a dictionary containing the model's generated outputs. The function extracts the first generated message content, transforms it to lowercase, removes trailing periods, commas, and spaces, and returns it as the summarized text.

#### 3.3.3 Performance and limitations
As described previously on the sentiment analysis section with GPT. It is note recommended (as far as may 2023) to make this process automatically due to Rate Limit Error in high peaks of use of OpenAI API. This could change in better models and improvements from OpenAI.

#### 3.3.4 References
Please consider this webpage as comparation of several summarization models including GPT-3.
* Summarization with several models [News Summarization comparison](https://tagoyal.github.io/zeroshot-news-annotations.html)



