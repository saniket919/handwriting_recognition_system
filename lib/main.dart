import 'package:flutter/material.dart';
import 'package:digit_recognition/recognizer_screen.dart';

void main() => runApp(HandwrittenNumberRecognizerApp());

class HandwrittenNumberRecognizerApp extends StatelessWidget {

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        title: 'Number Recognizer',
        theme: ThemeData(
          primarySwatch: Colors.grey,
        ),
      home: RecognizerScreen(title: 'Digit recogniser', key: null,),
    );
  }
}