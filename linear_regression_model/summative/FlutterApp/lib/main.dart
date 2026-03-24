import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Student Grade Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1E3A5F),
          primary: const Color(0xFF1E3A5F),
        ),
        useMaterial3: true,
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.grey.shade50,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: Color(0xFF1E3A5F), width: 2),
          ),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          labelStyle: const TextStyle(fontSize: 13),
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  // Replace with your Render URL after deployment
  static const String apiUrl = 'https://student-math-api.onrender.com/predict';

  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  String? _result;
  bool _isError = false;

  // -- Controllers (26 fields matching feature_columns.json) -----------------
  final _sex        = TextEditingController();
  final _age        = TextEditingController();
  final _address    = TextEditingController();
  final _famsize    = TextEditingController();
  final _Pstatus    = TextEditingController();
  final _Medu       = TextEditingController();
  final _Fedu       = TextEditingController();
  final _Mjob       = TextEditingController();
  final _reason     = TextEditingController();
  final _guardian   = TextEditingController();
  final _traveltime = TextEditingController();
  final _studytime  = TextEditingController();
  final _failures   = TextEditingController();
  final _schoolsup  = TextEditingController();
  final _paid       = TextEditingController();
  final _nursery    = TextEditingController();
  final _higher     = TextEditingController();
  final _internet   = TextEditingController();
  final _romantic   = TextEditingController();
  final _famrel     = TextEditingController();
  final _goout      = TextEditingController();
  final _Dalc       = TextEditingController();
  final _Walc       = TextEditingController();
  final _health     = TextEditingController();
  final _G1         = TextEditingController();
  final _G2         = TextEditingController();

  @override
  void dispose() {
    _sex.dispose(); _age.dispose(); _address.dispose();
    _famsize.dispose(); _Pstatus.dispose(); _Medu.dispose();
    _Fedu.dispose(); _Mjob.dispose(); _reason.dispose();
    _guardian.dispose(); _traveltime.dispose(); _studytime.dispose();
    _failures.dispose(); _schoolsup.dispose(); _paid.dispose();
    _nursery.dispose(); _higher.dispose(); _internet.dispose();
    _romantic.dispose(); _famrel.dispose(); _goout.dispose();
    _Dalc.dispose(); _Walc.dispose(); _health.dispose();
    _G1.dispose(); _G2.dispose();
    super.dispose();
  }

  // -- Field definitions (26 fields) ----------------------------------------
  List<Map<String, dynamic>> get _fields => [
    // Demographic
    {'label': 'Sex (0=Female, 1=Male)',          'ctrl': _sex,        'min': 0,  'max': 1},
    {'label': 'Age (15-22)',                      'ctrl': _age,        'min': 15, 'max': 22},
    {'label': 'Address (0=Rural, 1=Urban)',       'ctrl': _address,    'min': 0,  'max': 1},
    {'label': 'Family Size (0=LE3, 1=GT3)',       'ctrl': _famsize,    'min': 0,  'max': 1},
    {'label': 'Parent Status (0=Apart, 1=Tog)',   'ctrl': _Pstatus,    'min': 0,  'max': 1},
    // Parent info
    {'label': 'Mother Education (0-4)',           'ctrl': _Medu,       'min': 0,  'max': 4},
    {'label': 'Father Education (0-4)',           'ctrl': _Fedu,       'min': 0,  'max': 4},
    {'label': 'Mother Job (0-4)',                 'ctrl': _Mjob,       'min': 0,  'max': 4},
    // School
    {'label': 'Reason for School (0-3)',          'ctrl': _reason,     'min': 0,  'max': 3},
    {'label': 'Guardian (0-2)',                   'ctrl': _guardian,   'min': 0,  'max': 2},
    {'label': 'Travel Time (1-4)',                'ctrl': _traveltime, 'min': 1,  'max': 4},
    {'label': 'Study Time (1-4)',                 'ctrl': _studytime,  'min': 1,  'max': 4},
    {'label': 'Past Failures (0-4)',              'ctrl': _failures,   'min': 0,  'max': 4},
    // Support flags
    {'label': 'School Support (0=No, 1=Yes)',     'ctrl': _schoolsup,  'min': 0,  'max': 1},
    {'label': 'Paid Classes (0=No, 1=Yes)',       'ctrl': _paid,       'min': 0,  'max': 1},
    {'label': 'Nursery (0=No, 1=Yes)',            'ctrl': _nursery,    'min': 0,  'max': 1},
    {'label': 'Higher Ed Goal (0=No, 1=Yes)',     'ctrl': _higher,     'min': 0,  'max': 1},
    {'label': 'Internet Access (0=No, 1=Yes)',    'ctrl': _internet,   'min': 0,  'max': 1},
    {'label': 'Romantic (0=No, 1=Yes)',           'ctrl': _romantic,   'min': 0,  'max': 1},
    // Lifestyle
    {'label': 'Family Relations (1-5)',           'ctrl': _famrel,     'min': 1,  'max': 5},
    {'label': 'Going Out (1-5)',                  'ctrl': _goout,      'min': 1,  'max': 5},
    {'label': 'Weekday Alcohol (1-5)',            'ctrl': _Dalc,       'min': 1,  'max': 5},
    {'label': 'Weekend Alcohol (1-5)',            'ctrl': _Walc,       'min': 1,  'max': 5},
    {'label': 'Health Status (1-5)',              'ctrl': _health,     'min': 1,  'max': 5},
    // Academic
    {'label': 'G1 - First Period Grade (0-20)',   'ctrl': _G1,         'min': 0,  'max': 20},
    {'label': 'G2 - Second Period Grade (0-20)',  'ctrl': _G2,         'min': 0,  'max': 20},
  ];

  // -- Predict ---------------------------------------------------------------
  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() { _isLoading = true; _result = null; });

    try {
      final body = {
        'sex':        int.parse(_sex.text),
        'age':        int.parse(_age.text),
        'address':    int.parse(_address.text),
        'famsize':    int.parse(_famsize.text),
        'Pstatus':    int.parse(_Pstatus.text),
        'Medu':       int.parse(_Medu.text),
        'Fedu':       int.parse(_Fedu.text),
        'Mjob':       int.parse(_Mjob.text),
        'reason':     int.parse(_reason.text),
        'guardian':   int.parse(_guardian.text),
        'traveltime': int.parse(_traveltime.text),
        'studytime':  int.parse(_studytime.text),
        'failures':   int.parse(_failures.text),
        'schoolsup':  int.parse(_schoolsup.text),
        'paid':       int.parse(_paid.text),
        'nursery':    int.parse(_nursery.text),
        'higher':     int.parse(_higher.text),
        'internet':   int.parse(_internet.text),
        'romantic':   int.parse(_romantic.text),
        'famrel':     int.parse(_famrel.text),
        'goout':      int.parse(_goout.text),
        'Dalc':       int.parse(_Dalc.text),
        'Walc':       int.parse(_Walc.text),
        'health':     int.parse(_health.text),
        'G1':         int.parse(_G1.text),
        'G2':         int.parse(_G2.text),
      };

      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _isError = false;
          _result = 'Predicted G3: ${data['predicted_G3']} / 20\n${data['interpretation']}';
        });
      } else {
        final data = jsonDecode(response.body);
        setState(() {
          _isError = true;
          _result = 'Error: ${data['detail'] ?? response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _isError = true;
        _result = 'Connection error: $e';
      });
    } finally {
      setState(() { _isLoading = false; });
    }
  }

  // -- Build a single input field --------------------------------------------
  Widget _buildField(Map<String, dynamic> field) {
    final int min = field['min'];
    final int max = field['max'];
    final TextEditingController ctrl = field['ctrl'];
    final String label = field['label'];

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextFormField(
        controller: ctrl,
        keyboardType: TextInputType.number,
        decoration: InputDecoration(labelText: label),
        validator: (value) {
          if (value == null || value.isEmpty) return 'Required';
          final n = int.tryParse(value);
          if (n == null) return 'Must be a whole number';
          if (n < min || n > max) return 'Must be between $min and $max';
          return null;
        },
      ),
    );
  }

  // -- UI --------------------------------------------------------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E3A5F),
        foregroundColor: Colors.white,
        title: const Text(
          'Student Grade Predictor',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
        ),
        centerTitle: true,
      ),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Header card
            Card(
              color: const Color(0xFF1E3A5F),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
              child: const Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Final Math Grade Prediction',
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 4),
                    Text(
                      'Fill in all 26 fields to predict the student\'s G3 final math grade (0-20).',
                      style: TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            _sectionLabel('Demographic'),
            ..._fields.sublist(0, 5).map(_buildField),

            _sectionLabel('Parent Information'),
            ..._fields.sublist(5, 8).map(_buildField),

            _sectionLabel('School Information'),
            ..._fields.sublist(8, 13).map(_buildField),

            _sectionLabel('Support (0=No, 1=Yes)'),
            ..._fields.sublist(13, 19).map(_buildField),

            _sectionLabel('Lifestyle (1-5 scale)'),
            ..._fields.sublist(19, 24).map(_buildField),

            _sectionLabel('Academic Grades'),
            ..._fields.sublist(24, 26).map(_buildField),

            const SizedBox(height: 8),

            SizedBox(
              width: double.infinity,
              height: 52,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _predict,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1E3A5F),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  textStyle: const TextStyle(
                      fontSize: 16, fontWeight: FontWeight.bold),
                ),
                child: _isLoading
                    ? const SizedBox(
                    width: 24,
                    height: 24,
                        child: CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 2.5))
                    : const Text('Predict'),
              ),
            ),

            const SizedBox(height: 16),

            // Result display area
            if (_result != null)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: _isError ? Colors.red.shade50 : Colors.green.shade50,
                  border: Border.all(
                    color: _isError
                        ? Colors.red.shade300
                        : Colors.green.shade300,
                  ),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _isError ? 'Prediction Failed' : 'Prediction Result',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                        color: _isError
                            ? Colors.red.shade700
                            : Colors.green.shade700,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _result!,
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: _isError
                            ? Colors.red.shade800
                            : const Color(0xFF1E3A5F),
                      ),
                    ),
                  ],
                ),
              ),

            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _sectionLabel(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10, top: 4),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.bold,
          color: Color(0xFF1E3A5F),
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
