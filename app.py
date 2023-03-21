import re
import pickle
from flask import Flask, jsonify, request
from happytransformer import HappyGeneration, GENSettings

# load the saved model using Pickle
model_weights_path = "happy_gen_model_weights.pkl"

with open(model_weights_path, 'rb') as f:
    model_weights = pickle.load(f)

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")
happy_gen.model.load_state_dict(model_weights)

training_cases = """Keywords: Canada, AI, fast
Output: Canada's AI industry is growing fast. 
###
Keywords: purchase, desk, adjustable
Output: I just purchased a new height adjustable desk. 
###
Keywords: museum, art, painting, Ottawa
Output: I went to an art museum in Ottawa and saw some beautiful paintings. I'm excited to revisit. 
###
Keywords: exam, success, study
Output:My first exam was a success! I think I aced it because of your help with studying. 
###
Keywords: Self Programmable,medical monitoring device,Can adapt it's algorithms to each patient,Uploads algorithms,Can re- configure for other applications, such as: EMG, ECG, EEG,Can work Stand-alone,Or Work with outside computers,Wireless Communication
Output:A network system managing an on-demand service within a geographic region can receive, over a network, multi-user request data corresponding to a request for service for a plurality of users. The request data can indicate one or more start locations, a plurality of users, and one or more service locations. In response to receiving the request data, the network system can select a set of service providers from a plurality of candidate service providers in the geographic region to provide the requested service. The service providers can be selected based on optimizations of one or more service parameters including estimated fares for the plurality of users, ETAs to the start location, ETAs to the service locations, etc. The network system can further determine routes for the set of service providers from their respective locations to the start or service location(s) and from the start or 
###
Keywords:Portable device,Wireless connectivity,A Security device,With method of use,Provides multi-factor authentication,For online access,using proximity with mobile devices as authentication,Use biometric for authentication factor
Output:A portable security device with wireless connectivity that provides multi-factor authentication for online access. It has a built-in method of use that allows users to authenticate their identity using biometric factors. The device also uses proximity with mobile devices as an additional authentication factor, making it more difficult for unauthorized users to access the device. Overall, the device is designed to provide a high level of security for online access and is particularly well-suited for users who need to access sensitive information or systems remotely.
###
Keywords:   An apparatus,For storing pills,and dispensing pills,With cartridge mechanism,Has optical sensors,Non-contact sensors
Output:An apparatus is designed to provide easy and efficient access to pills, while also ensuring accuracy and reliability. The device uses non-contact sensors to detect the presence or absence of pills in the cartridge and optical sensors to ensure that the correct number of pills are dispensed. The cartridge mechanism allows for the quick and easy replacement of pill supplies, making it a practical and convenient option for individuals who require regular medication. Overall, the device is designed to simplify the process of managing medication and ensure that individuals have easy and reliable access to the pills they need.
###
Keywords:System and Method,Improving remote patient monitoring,Using plurality of physiological medical devices,Wireless devices,Includes a Virtual assistant Avatar,Includes physician web portal,Remote testing of patient,Uses Artificial Intelligence,Utilizing speech to text, Text to speech and Natural language processing
Output:A wireless devices that include a virtual assistant avatar, which is capable of communicating with patients in a natural and intuitive manner. The system also includes a physician web portal, which allows healthcare professionals to remotely monitor patient data and provide feedback and recommendations. The virtual assistant avatar provides patients with an easy and interactive way to report symptoms and respond to questions, while the physician web portal allows doctors to remotely test patients and monitor their progress. The system uses artificial intelligence to analyze patient data and provide insights and recommendations to healthcare professionals, while also utilizing speech to text, text to speech, and natural language processing to ensure effective communication between patients and the virtual assistant avatar. Overall, the system and method is designed to improve remote patient monitoring, provide more efficient and effective communication between patients and healthcare professionals, and facilitate better overall patient outcomes.
###
Keywords:Smart Medication tray,Wireless connectivity,Built-in weight scales,Medication sensors,With storage compartments,With proximity sensors,Biometric access
Output:a wireless device that includes built-in weight scales, medication sensors, storage compartments, proximity sensors, and biometric access. The device is designed to help patients manage their medications more effectively and efficiently. The built-in weight scales ensure that patients are taking the correct dosage of their medication, while the medication sensors ensure that the medication is taken at the right time. The storage compartments are designed to store different types of medication separately, reducing the risk of confusion or errors. The proximity sensors allow the device to detect when the patient is nearby and remind them to take their medication, while the biometric access ensures that only authorized users have access to the medication tray. Overall, the smart medication tray is an innovative solution that helps patients manage their medication more effectively, while also providing a higher level of security and reliability.
###"""

# Initializing the flask app
app = Flask(__name__)
# Define your Flask API routes
@app.route("/generate", methods=["POST"])
def generate_text():
     # Get input data from request
    data = request.json
    keywords = data['keywords']

    # Check format of keywords and convert to list
    if isinstance(keywords, str):
        # Check if keywords are comma-separated
        if "," in keywords:
            keywords_list = keywords.split(",")
        else:
            # Check if keywords are bulleted list
            keywords_list = re.findall(r"[\*•]\s*(\w+)", keywords)
    elif isinstance(keywords, list):
        keywords_list = keywords
    else:
        return jsonify({'error': 'Invalid format for keywords'})
    
    #Creating the prompt
    def create_prompt(training_cases, keywords):
        keywords = ", ".join(keywords)
        prompt = training_cases + "\nKeywords: "+ keywords+ "\nOutput:"
        return prompt

    prompt = create_prompt(training_cases, keywords)

    # Generate text using the HappyGeneration model
    args_beam = GENSettings(num_beams=5, no_repeat_ngram_size=3, early_stopping=True, min_length=1, max_length=100)
    result = happy_gen.generate_text_from_model(prompt, args=args_beam)

    # Generate text using the HappyGeneration model
    output_text = happy_gen.generate_text(prompt, args=args_beam)

    # Return the generated text as a JSON response
    return jsonify({"output_text": output_text})

if __name__ == "__main__":
    app.run(debug=True)
