import React from 'react';
import { render, screen } from '@testing-library/react';
import { Formik } from 'formik';
import FormField from '../FormField';

describe('FormField', () => {
  it('renders label and input correctly', () => {
    render(
      <Formik initialValues={{ test: '' }} onSubmit={() => {}}>
        <FormField name="test" label="Test Field" />
      </Formik>
    );
    expect(screen.getByLabelText('Test Field')).toBeInTheDocument();
  });
}); 