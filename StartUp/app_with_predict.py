#############################################
# Kullanılacak Kütüphaneler
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_extras.let_it_rain import rain
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
#import matplotlib.pyplot as plt
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator DecisionTreeRegressor*")
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator DummyClassifier*")
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator GradientBoostingClassifier*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="In a future version, `df.iloc[:, i] = newvals`*")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
# Disable the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # 'warn' is the default mode
# Re-enable the SettingWithCopyWarning (if needed)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import StandardScaler
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
from transformers import pipeline



from streamlit_option_menu import option_menu

#############################################
# Sayfaların Oluşumu
#############################################

st.set_page_config(
    page_title='DataStrophic2',
    layout="wide",
    page_icon="💰")

page_bg_img = """
<style>
[data-testid="stSidebar"] {
background-image: url("https://i.pinimg.com/originals/9f/73/95/9f73957ced55190e40212bf1da84dc92.jpg");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu("StartUp Project ", ["Predict", "Report",  "---", 'Communication'],
                           icons=['house',  None, 'cloud-upload'], menu_icon="cast",
                           default_index=0)


#############################################
# Predict Sayfası
#############################################


if selected == "Predict":
    st.title("StartUp Prediction")


    # Modeli yükleyin
    # Modelin tam yolunu kullanarak modeli yükleyin
    #model_path = 'pickle_clean_crunchbase.pkl'
    #model = pickle.load(model_path)

    with open('x_test.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('pickle_clean_crunchbase.pkl', 'rb') as model_file:
        display_df = pickle.load(model_file)


    st.write("Let's preview dataset ...")

    st.write(display_df[['country', 'recency', 'funding_total_usd', 'seed', 'venture',
                           'round_A', 'diff_funding_months', 'diff_first_funding_months',
                           'round_A_H_total', 'avg_fund_size']].head())


    # Kullanıcıdan girdi alacağız
    # Özellikleri belirtin ve değerlerini isteyin

    st.sidebar.title("Startup Funding values")

    # Collects user input features into dataframe
    #uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    #if uploaded_file is not None:
        #input_df = pd.read_csv(uploaded_file)
    def user_input_features():
        country_mm = st.sidebar.selectbox("country", ( 'Turkey', 'United States', 'United Kingdom',
                                                   'Canada', 'China', 'Germany', 'France','India',
                                                   'Israel', 'Spain', 'Russia', 'Sweden', 'Italy', 'Netherlands',
                                                   'Ireland', 'Singapure', 'Norway', 'Poland'))
        funding_total_usd_mm = st.sidebar.slider('funding_total_usd', 0, 100000, 1000000000)
        seed_mm = st.sidebar.slider('seed', 0, 100000, 10000000)
        venture_mm = st.sidebar.number_input('venture', step=1)
        round_A_mm = st.sidebar.number_input('round_A', step=1)
        founded_mm = st.sidebar.date_input("founded")   #burası date aralığı olabilir ?
        first_funded_mm = st.sidebar.date_input("first_funded" ) # https://docs.streamlit.io/library/api-reference/widgets/st.date_input buradan yapılabilir
        last_funded_mm = st.sidebar.date_input("last_funded" ) # https://docs.streamlit.io/library/api-reference/widgets/st.date_input buradan yapılabilir
        round_A_H_total_mm = st.sidebar.number_input('round_A_H_total', step=1)
        avg_fund_size_mm = st.sidebar.slider("avg_fund_size", 0, 10000, 15000000)

        feature_list = [country_mm,funding_total_usd_mm,seed_mm,venture_mm,round_A_mm,founded_mm,
                        first_funded_mm,last_funded_mm,round_A_H_total_mm,avg_fund_size_mm]
        #market_mm = st.sidebar.selectbox('market', ('Publishing','Electronics','Tourism'
        #                                             'Software','Biotechnology','Education'))
        #status_mm = st.sidebar.radio("Status",('Acquired', 'Closed', 'Operating'))
        #equity_crowdfunding_mm = st.sidebar.radio("equity_crowdfunding",('Acquired', 'Closed', 'Operating'))
        #undisclosed_mm = st.sidebar.radio("undisclosed",('Acquired', 'Closed', 'Operating'))
        #convertible_note_mm = st.sidebar.radio("convertible_note",('Acquired', 'Closed', 'Operating'))
        #debt_financing_mm = st.sidebar.radio("debt_financing",('Acquired', 'Closed', 'Operating'))
        #angel_mm = st.sidebar.radio("angel", ('Acquired', 'Closed', 'Operating'))
        #grant_mm = st.sidebar.radio("grant", ('Acquired', 'Closed', 'Operating'))
        #private_equity_mm = st.sidebar.radio("private_equity", ('Acquired', 'Closed', 'Operating'))
        #post_ipo_equity_mm = st.sidebar.radio("post_ipo_equity", ('Acquired', 'Closed', 'Operating'))
        #post_ipo_debt_mm = st.sidebar.radio("post_ipo_debt", ('Acquired', 'Closed', 'Operating'))
        #secondary_market_mm = st.sidebar.radio("secondary_market", ('Acquired', 'Closed', 'Operating'))
        #product_crowdfunding_mm = st.sidebar.radio("product_crowdfunding", ('Acquired', 'Closed', 'Operating'))
        #round_B_mm = st.sidebar.radio("round_B", ('Acquired', 'Closed', 'Operating'))
        #round_C_mm = st.sidebar.radio("round_C", ('Acquired', 'Closed', 'Operating'))
        #round_D_mm = st.sidebar.radio("round_D", ('Acquired', 'Closed', 'Operating'))
        #round_E_mm = st.sidebar.radio("round_E", ('Acquired', 'Closed', 'Operating'))
        #round_F_mm = st.sidebar.radio("round_F", ('Acquired', 'Closed', 'Operating'))
        #round_G_mm = st.sidebar.radio("round_G", ('Acquired', 'Closed', 'Operating'))
        #round_H_mm = st.sidebar.radio("round_H", ('Acquired', 'Closed', 'Operating'))
        #angel_status_mm = st.sidebar.radio("angel_status", ('Acquired', 'Closed', 'Operating'))
        #grant_status_mm = st.sidebar.radio("grant_status", ('Acquired', 'Closed', 'Operating'))
        #ratio_seed_tot_mm = st.sidebar.radio("ratio_seed_tot", ('Acquired', 'Closed', 'Operating'))
        #ratio_debt_tot_mm = st.sidebar.radio("ratio_debt_tot", ('Acquired', 'Closed', 'Operating'))
        #convertible_status_mm = st.sidebar.radio("convertible_status", ('Acquired', 'Closed', 'Operating'))
        #seed_quartiles_mm = st.sidebar.radio("seed_quartiles", ('Acquired', 'Closed', 'Operating'))
        #angel_degree_mm = st.sidebar.radio("angel_degree", ('Acquired', 'Closed', 'Operating'))
        #tot_funding_degree_mm = st.sidebar.radio("tot_funding_degree", ('Acquired', 'Closed', 'Operating'))
        #venture_degree_mm = st.sidebar.radio("venture_degree", ('Acquired', 'Closed', 'Operating'))
        #start_postion_mm = st.sidebar.radio("start_postion", ('Acquired', 'Closed', 'Operating'))
        #secondary_status_mm = st.sidebar.radio("secondary_status", ('Acquired', 'Closed', 'Operating'))


        data = {#'market': market_mm,
                'funding_total_usd': funding_total_usd_mm,
                #'status': status_mm,
                'seed': seed_mm,
                'venture': venture_mm,
                #'equity_crowdfunding': equity_crowdfunding_mm,
                #'undisclosed': undisclosed_mm,
                #'convertible_note': convertible_note_mm,
                #'debt_financing': debt_financing_mm,
                #'angel': angel_mm,
                #'grant': grant_mm,
                #'private_equity': private_equity_mm,
                #'post_ipo_equity': post_ipo_equity_mm,
                #'post_ipo_debt': post_ipo_debt_mm,
                #'secondary_market': secondary_market_mm,
                #'product_crowdfunding': product_crowdfunding_mm,
                'round_A': round_A_mm,
                #'round_B': round_B_mm,
                #'round_C': round_C_mm,
                #'round_D': round_D_mm,
                #'round_E': round_E_mm,
                #'round_F': round_F_mm,
                #'round_G': round_G_mm,
                #'round_H': round_H_mm,
                'country': country_mm,
                'founded': founded_mm,
                'first_funded': first_funded_mm,
                'last_funded_mm' : last_funded_mm,
                'round_A_H_total': round_A_H_total_mm,
                #'angel_status': angel_status_mm,
                #'grant_status': grant_status_mm,
                'avg_fund_size': avg_fund_size_mm,
                #'ratio_seed_tot': ratio_seed_tot_mm,
                #'ratio_debt_tot': ratio_debt_tot_mm,
                #'convertible_status': convertible_status_mm,
                #'seed_quartiles': seed_quartiles_mm,
                #'angel_degree': angel_degree_mm,
                #'tot_funding_degree': tot_funding_degree_mm,
                #'venture_degree': venture_degree_mm,
                #'start_postion': start_postion_mm,
                #'secondary_status': secondary_status_mm,
                }

        features = pd.DataFrame(data, index=[0])

        return features,feature_list

    st.title("What is your great Ai Startup idea ? ")
    model_checkpoint = "consciousAI/question-answering-roberta-base-s-v2"


    context = """  atomic.vc  Atomic is a venture studio: we prototype new companies and assemble teams to scale the most promising ideas into independent ventures. We're hiring for a portfolio of startups looking for engineers, designers, product managers, and more.",
 "greylock  We've worked with Figma, Abnormal Security, Instabase, and many other $B+ companies starting in their stealth days. Hiring for a portfolio of startups. We would love to connect with individuals looking to explore or join as engineers, product managers, and designers",
 'cynch.ai  Building AI for accounting and financial decision making',
 'synthesia.io  Building AI avatars that generate professional videos in minutes',
 'openai  OpenAI is a nonprofit AI research company, discovering and enacting the path to safe artificial general intelligence.',
 'memorahealth  Helping healthcare orgs simplify and automate care journeys',
 'tavus.io  Use your voice for sales, without ever saying a word',
 'replicate  Run machine learning models in the cloud',
 'mindsdb  Help anyone use the power of ML to ask predictive questions of their data and receive accurate answers',
 'madstreetden  Mad Street Den is a computer vision and artificial intelligence startup.',
  'deepl  DeepL is a leading deep learning company for language translation',
 'shieldai  Develop AI products for the protection of service members and civilians',
 'spot.ai  Create safer, smarter organizations with your new AI Camera System',
 'fathomhealth  Medical coding automation powered by AI'
   """



    question = st.text_input('Enter your idea :',"...")

    # question = "Which company developes labeled-data ? "
    # #question = " I want to work with ai and customer service which tool can help me ? "

    question_answerer = pipeline("question-answering", model=model_checkpoint)
    result = question_answerer(question=question, context=context)


    fikir_button = st.button("Did someone think about it ?")
    if fikir_button == True:
        st.write(result["answer"])





    st.markdown("Entered values : ")

    input_df,input_variables = user_input_features()
    # input variables bir list burada girilen değerleri alıp kullanıyor.

    #    st.dataframe(input_df[['country', 'recency', 'funding_total_usd', 'seed', 'venture',
    #  'round_A', 'diff_funding_months', 'diff_first_funding_months',
    #                      'round_A_H_total', 'avg_fund_size']])


    ##########################################################################################
    ## HERE WE'LL BE PREDICTNG THROUGH LOADING MODEL
    ##########################################################################################
    pd.options.mode.chained_assignment = None
    # modeli yükllüyorum
    with open('final_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open('model_ready_ds.pkl', 'rb') as model_file:
        ml_df = pickle.load(model_file)


    a = 1800
    example = ml_df.iloc[a:a + 1]
    example_actual_result = example['target']
    example = example.drop('target', axis=1)
    # new example is going to be modified


    input_df['diff_first_funding_months'] = (input_df['first_funded'] - input_df['founded']) / np.timedelta64(1, 'M')
    input_df['diff_funding_months'] = (input_df['last_funded_mm'] - input_df['first_funded']) / np.timedelta64(1, 'M')
    max_last_fund_date = input_df['last_funded_mm'].max()
    input_df['recency'] = (max_last_fund_date - input_df["founded"]) / np.timedelta64(1, 'M')


    #['recency','diff_first_funding_months','diff_funding_months','funding_total_usd','venture',
    # 'avg_fund_size','country_United States','round_A_H_total','seed','round_A']






    #for each in features_to_enter:
    #    example[each] = input_df[each]
    # -108 is because selected date is current but data set is old diff in months

    example['recency'] = float(input_df['recency'])
    example['diff_first_funding_months'] = float(input_df['diff_first_funding_months']-108)
    example['diff_funding_months'] = float(input_df['diff_funding_months']-108)
    example['funding_total_usd'] = int(input_df['funding_total_usd'])
    example['venture'] = float(input_df['venture'])
    example['avg_fund_size'] = float(input_df['avg_fund_size'])
    example['country_United States'] = False
    example['round_A_H_total'] = int(input_df['round_A_H_total'])
    example['seed'] = float(input_df['seed'])
    example['round_A'] = float(input_df['round_A'])

    # ac, pre = predict_single_sample(example)

    pre = loaded_model.predict(example)

    st.dataframe(input_df)

    button = st.button("How my startup will perform ? ")
    if button == True:
        st.title("Your startup : ")
        if pre == 0:
            out_str = " Probably will be closed :( You can focus on another one"
        elif pre == 1:
            out_str = " Probably will be acquired CONGRATS :) You will have an exit "
        else:
            out_str = " Keep working hard ! Yes you will survive "
        st.write(out_str)






#############################################
# Analiz Sayfası
#############################################

# Bu kısmı kaldırdık


#############################################
# Raporlama Sayfası
#############################################

if selected == "Report":


    st.title("Reports and Analysis for Chrunchbase Dataset")
    st.write("You can find brief summary statistics and graphs ")



    with open('pickle_clean_crunchbase.pkl', 'rb') as model_file:
        display_df = pickle.load(model_file)

        report_df = display_df[display_df["status"]=='acquired']



    display_df[display_df["status"] == 'acquired'].shape
    st.write("### Dataset:")
    st.write(report_df.head())

    show_profile_report = st.checkbox("Show Reports")

    if show_profile_report:
        st.write("### Pandas Profiling Report:")
        profile = ProfileReport(report_df, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile)

        # Veri kümesinin özellik dağılımlarını göster
        st.write("### Feature Distributions:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=report_df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Sütunlar arasındaki ilişkileri göster
        st.write("### Column Interactions:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pairplot(data=report_df, diag_kind="kde")
        st.pyplot(fig)

        # Korelasyon matrisini göster
        st.write("### Correlation Matrix:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(report_df.corr(), annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
        st.pyplot(fig)


#############################################
# LLM Sayfası
#############################################

if selected == "Large Language M":

    model_checkpoint = "consciousAI/question-answering-roberta-base-s-v2"



    #st.title("Is there such company ? ")
    #context = """
    #atomic.vc  Atomic is a venture studio: we prototype new companies and assemble teams to scale the most promising ideas into independent ventures. We re hiring for a portfolio of startups looking for engineers, designers, product managers, and more.", "greylock  We ve worked with Figma, Abnormal Security, Instabase, and many other $B+ companies starting in their stealth days. Hiring for a portfolio of startups. We would love to connect with individuals looking to explore or join as engineers, product managers, and designers" cynch.ai  Building AI for accounting and financial decision making  synthesia.io  Building AI avatars that generate professional videos in minutes  openai  OpenAI is a nonprofit AI research company, discovering and enacting the path to safe artificial general intelligence  memorahealth  Helping healthcare orgs simplify and automate care journeys  tavus.io  Use your voice for sales, without ever saying a word  replicate  Run machine learning models in the cloud  mindsdb  Help anyone use the power of ML to ask predictive questions of their data and receive accurate answers  madstreetden  Mad Street Den is a computer vision and artificial intelligence startup  deepl  DeepL is a leading deep learning company for language translation  shieldai  Develop AI products for the protection of service members and civilians  spot.ai  Create safer, smarter organizations with your new AI Camera System  fathomhealth  Medical coding automation powered by AI  deepgram  Provide developers with a simple to use Speech-to-Text API  amprobotics  Sort recyclable material at a fraction of the cost of current technology  golden  Developer of a self-constructing knowledge database used to accelerate discovery and education  kumo.ai  Kumo allows businesses to make faster, simpler, and smarter predictions  poly-ai  PolyAI develops a machine learning platform for conversational artificial intelligence  abridge  Build audio-based system to record and summarize medical conversations  aisera  AISERA provides AI Service Experience solutions for enterprises to automate IT tasks and workflows with conversational AI and RPA  marqvision  Helps brands remove counterfeits from online marketplaces through AI-powered platform  assemblyai  Building #1 rated API to transcribe and understand audio data  tecton.ai  Tecton provides an enterprise-ready feature store to make machine learning accessible to every company  optimaldynamics  Using AI and ML to transform the logistics industry  builtrobotics  Built Robotics develops automated guidance systems for the 1 trillion heavy equipment industry  viz.ai  Viz is a medical imaging company that helps optimize emergency treatment using deep learning technology  lilt  Lilt is the modern language service and technology provider enabling localized customer experiences  cresta  Cresta leverages artificial intelligence to help sales and service agents improve the quality of their customer service  deepcellbio  Deepcell uses AI to isolate and capture cells based on morphological features, for multiple research and translational applications  d-id  Turning images & videos into Creative Reality experiences  salt.security  Salt Security provides an API protection platform designed to prevent attacks by leveraging machine learning and AI  veriff.me  Veriff is a global tech company building a visionary AI driven verification platform  labelbox  Labelbox creates and manages labeled data for machine learning applications  hyperscience  Hyperscience develops AI-based enterprise software designed to automate office work processes  robustintelligence  Eliminate AI failures for companies by stress-testing their models at the click of a button  robinhealthcare  AI medical scribe that takes care of medical documentation  getzuma  Building 24/7 human + AI sales teams, starting with real estate property management  socure  Socure is a predictive analytics platform for digital identity verification of consumers  vergesense  VergeSense is a software-as-a-service company that develops artificial intellegence-powered workplace sensors  splashhq  Splash is helping everyone make music  shift-technology  Shift Technology provides AI-driven decision automation and optimization for the global insurance industry  copy.ai  CopyAI is building AI-powered copywriting tools for business customers  neuralmagic  Machine learning without limits  osaro  Osaro is an AI company developing products based on proprietary deep reinforcement learning technology  facilio  Facilio is a management software that employs IoT and machine learning to help manage buildings  heartlab.ai  Transforming cardiology with artificial intelligence  catchandrelease  Catch&Release is a content licensing platform that enables brands and advertisers to license content from anywhere  centaurlabs  Labeling medical images at scale  moveworks  Moveworks is a cloud-based AI platform  neuralink  Developing ultra high bandwidth brain-machine interfaces to connect humans and computers  hioperator  HiOperstor provides customer support as-a-service  playment.io  We are fully-managed solution offering training data for computer vision, machine learning and human-in-the-loop for AI at scale  botmd.io  Your friendly A.I. clinical assistant. I provide doctors with a smarter, simpler way to search clinical content  standard.ai  Provides an autonomous checkout tool that can be installed into retailers’ existing stores , "phiar.net  Phiar s Ultra-lightweight Spatial AI Engine enables vehicles to perceive its surroundings, and powers our first use case in AR Navigation." crowdai  CrowdAI empowers organizations to create and deploy custom models for visual AI, in a simple, code-free platform  getjerry  Jerry is an AI, ML, and bot-powered car ownership app that saves customers time and money on car expenses  scale  Scale AI is the data platform for AI, providing high quality training data for leading machine learning teams  crosschq  Crosschq is a technology platform that provides a human intelligence-based hiring platform  ada.cx  Ada is an automated customer experience company that provides chat bots used in customer support  aeye.ai  AEye develops advanced vision hardware, software and algorithms that act as the eyes and visual cortex of autonomous vehicles  databricks  Databricks is a data-and-AI company that interacts with corporate information stored in the public cloud  people.ai  People is an AI platform for enterprise sales, marketing, and customer success that uncovers revenue opportunities  sisu.ai  Sisu is a decision intelligence engine that delivers insights and monitors metrics for businesses  dialpad  Dialpad is an AI-powered communications platform that turns conversations into opportunities and helps teams make smart calls  akasa  AKASA is an AI-powered automation company for revenue cycle management in healthcare  dynotx  Dyno Therapeutics uses artificial intelligence for gene therapy  capeanalytics  Cape Analytics provides AI and analytics services for properties  vectra.ai  Vectra is a cybersecurity platform that uses AI to detect attackers in real time and perform conclusive incident investigations  realityengines.ai  Abacus.AI is an artificial intelligence research and AI cloud services company  vahan.ai  We use Artificial Intelligence to bring job services inside WhatsApp  deepgenomics  Deep Genomics is using artificial intelligence to build a new universe of life-saving genetic therapies  tempo.fit  Tempo is a home fitness platform, combining equipment, training, and social motivation with 3D sensors and artificial intelligence  docbot.ai  Docbot, Inc. is an artificial intelligence technology company focusing on gastrointestinal (GI) disease  lexion.ai  Lexion is an AI-powered contract management startup  orbitalinsight  Builds SaaS technology to understand what happens on and to the Earth with AI and machine learning  tessian  Tessian creates a Human Layer Security that helps people work without security disruptions getting in the way  DominoDataLab  Domino Data Lab utilizes data science and AI for collaboration, model deployment, and centralizing infrastructure  uipath  UiPath is a leading provider of software automation and robotic process automation software  bigeye  Automatic data quality monitoring for analytics and data engineering teams  vise  Vise is an AI-driven portfolio management platform for financial advisors  physna  Physna is a geometric deep-learning and 3D search company that searches, compares, and analyzes 3D models  callsign  Delivering digital trust through simple secure customer interactions  mihup  Intelligence-based personal mobile assistant app  revl  AI Powered Video Souvenirs as a Service. Full end to end solution for roller coasters, skydive drop zones, race tracks, and zip lines  cureskin  Cureskin is an AI powered application that provides derma care through mobile devices  observe.ai  Observe.AI develops a voice AI platform for contact centers, turning agents into top performers  aifoundation  The AI Foundation’s mission is to move humanity forward through the power of decentralized, trusted, personal AI  sisense  Sisense is the leading AI-driven platform for infusing analytics everywhere  irisonboard  We are building revolutionary new ways for drones and unmanned systems to see and navigate the world  pixielabs.ai  We build machine intelligence systems which empower developers to engineer the future  rigetti  Rigetti Computing is a full-stack quantum computing company that designs and manufactures integrated circuits  drishti  Drishti is a provider of AI-powered video analytics technology that gives visibility and insights for manual assembly line improvement  rasa  Rasa is the leading open source machine learning toolkit that lets developers build conversational bots  bidgely  Bidgely tells you how much energy your appliances use and crafts personalized recommendations to save  atomwise  Atomwise develops artificial intelligence systems using powerful deep learning algorithms and supercomputers for drug discovery  captionhealth  Caption Health uses AI to interpret ultrasound exams  curaihealth  Curai Health is a virtual care company that uses AI to provide chat-based primary care at a lower cost  graphcore.ai  Graphcore is the inventor of the Intelligence Processing Unit (IPU), a microprocessor designed for AI and machine learning applications  kneron  Kneron develops an application-specific integrated circuit and software that offers artificial intelligence-based tools  pony.ai  Pony.ai is a developer of AI-based robot designed for autonomous driving  allenai.org  AI2 is an artificial intelligence research institute and start-up incubator  intelecy  Norwegian tech company using machine learning to prevent breakdowns, predict failures, improve production processes  bluehexagon.ai  Blue Hexagon offers an on-device machine learning-based malware detection  soundhound  SoundHound is the innovator in voice-enabled AI and conversational intelligence technologies  pindrop  Pindrop uses AI-based Authentication and Anti-Fraud Solutions to increase efficiency in call centers and stop fraudulent transactions  voxelcloud.io  VoxelCloud provides medical image analysis and diagnosis assistance based on AI and cloud computing technologies  sourceress  Sourceress is an AI recruiter that is reinventing how people find jobs  cataliahealth  We create the Mabu personal healthcare companion to help patients with chronic disease and collect data to healthcare providers  replika.ai  AI companion who cares  acalvio  Acalvio provides Advanced Threat Defense solutions to detect, engage, and respond to malicious activity inside the perimeter  sourceress  Sourceress is an AI recruiter that is reinventing how people find jobs  cataliahealth  We create the Mabu personal healthcare companion to help patients with chronic disease and collect data to healthcare providers  replika.ai  AI companion who cares  acalvio  Acalvio provides Advanced Threat Defense solutions to detect, engage, and respond to malicious activity inside the perimeter
    #"""
    input = st.text_input('Enter your idea :')
    #question = input
    # #question = " I want to work with ai and customer service which tool can help me ? "

    #question_answerer = pipeline("question-answering", model=model_checkpoint)
    #result = question_answerer(question=question, context=context)
    st.write(input)
#############################################
# Communication Sayfası
#############################################

if selected == "Communication":
    #st.balloons()

    rain(
        emoji="🎈",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",
    )

    st.markdown('<h1 style="text-align: center;">🤖 DataStrophic Team 🤖</h1>', unsafe_allow_html=True)
    #st.title("🤖 DataStrophic Team 🤖 ")

    col1, col2, = st.columns(2)

    with col1:
        st.image("Feyza.png")
        st.markdown('<h3 style="text-align: left;">Feyza Kamber</h3>', unsafe_allow_html=True)

        link_lIn = "[LinkedIn](https://www.linkedin.com/in/feyza-kamber-a61946a8/)"
        st.markdown(link_lIn, unsafe_allow_html=True)

        link_git = "[Github](https://github.com/FKamber)"
        st.markdown(link_git, unsafe_allow_html=True)

        link_kaggle = "[Kaggle](https://www.kaggle.com/feyzakamber)"
        st.markdown(link_kaggle, unsafe_allow_html=True)

        link_medium = "[Medium](https://medium.com/@kamberfeyza)"
        st.markdown(link_medium, unsafe_allow_html=True)

    with col2:
        st.image("Arda.jpeg")
        st.markdown('<h3 style="text-align: left;">Arda Asut</h3>', unsafe_allow_html=True)

        link_lIn = "[LinkedIn](https://www.linkedin.com/in/arda-asut-109800172/)"
        st.markdown(link_lIn, unsafe_allow_html=True)

        link_git = "[Github](https://github.com/ardoktor)"
        st.markdown(link_git, unsafe_allow_html=True)

        link_kaggle = "[Kaggle](https://www.kaggle.com/ardoktor)"
        st.markdown(link_kaggle, unsafe_allow_html=True)

