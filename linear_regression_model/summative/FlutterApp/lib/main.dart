import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const StudentMathPredictorApp());
}

class StudentMathPredictorApp extends StatelessWidget {
  const StudentMathPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Student Math Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0E7490)),
        useMaterial3: true,
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
  static const String apiBaseUrl = 'https://your-render-service.onrender.com';

  final Map<String, TextEditingController> controllers = {
    'age': TextEditingController(),
    'Medu': TextEditingController(),
    'Fedu': TextEditingController(),
    'traveltime': TextEditingController(),
    'studytime': TextEditingController(),
    'failures': TextEditingController(),
    'famrel': TextEditingController(),
    'freetime': TextEditingController(),
    'goout': TextEditingController(),
    'Dalc': TextEditingController(),
    'Walc': TextEditingController(),
    'health': TextEditingController(),
    'absences': TextEditingController(),
    'G1': TextEditingController(),
    'G2': TextEditingController(),
  };

  bool isLoading = false;
  String resultMessage = 'Prediction result will appear here.';

  @override
  void dispose() {
    for (final c in controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> predict() async {
    setState(() {
      isLoading = true;
      resultMessage = 'Predicting...';
    });

    try {
      final Map<String, dynamic> payload = {};
      for (final entry in controllers.entries) {
        final value = entry.value.text.trim();
        if (value.isEmpty) {
          throw const FormatException('All fields are required.');
        }
        payload[entry.key] = int.parse(value);
      }

      final uri = Uri.parse('$apiBaseUrl/predict');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          resultMessage =
              'Predicted G3: ${body['predicted_g3']} (Model: ${body['selected_model']})';
        });
      } else {
        final body = jsonDecode(response.body);
        setState(() {
          resultMessage =
              'Error ${response.statusCode}: ${body is Map<String, dynamic> ? body['detail'] : response.body}';
        });
      }
    } on FormatException catch (e) {
      setState(() {
        resultMessage = 'Input error: ${e.message}';
      });
    } catch (e) {
      setState(() {
        resultMessage = 'Request failed: $e';
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final labels = controllers.keys.toList();

    return Scaffold(
      appBar: AppBar(title: const Text('Student Math Grade Predictor')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
              child: ListView.builder(
                itemCount: labels.length,
                itemBuilder: (context, index) {
                  final key = labels[index];
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: TextField(
                      controller: controllers[key],
                      keyboardType: TextInputType.number,
                      decoration: InputDecoration(
                        labelText: key,
                        border: const OutlineInputBorder(),
                      ),
                    ),
                  );
                },
              ),
            ),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: isLoading ? null : predict,
                child: Text(isLoading ? 'Predicting...' : 'Predict'),
              ),
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.teal.shade50,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(resultMessage),
            ),
          ],
        ),
      ),
    );
  }
}
