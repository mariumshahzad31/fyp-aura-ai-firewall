from utils.prediction import AuraPredictor

p = AuraPredictor()
result = p.predict_records([{
    'Data': 'CVE-2023-1234',
    'cvss': 7.5,
    'Firewall Traffics': '192.168.1.1 -> 10.0.0.1',
    'cwe_code': 'CWE-79',
    'cwe_name': 'Cross-site Scripting',
    'summary': 'Test vulnerability'
}])
print('Prediction successful:', result[0]['risk_class'])