import React from 'react';
import { Field, ErrorMessage } from 'formik';
import clsx from 'clsx';

interface FormFieldProps {
  name: string;
  label: string;
  type?: string;
  placeholder?: string;
  options?: { value: string; label: string }[];
  min?: string;
  max?: string;
  step?: string;
  rows?: number;
  autoComplete?: string;
  className?: string;
  multiple?: boolean;
}

const FormField: React.FC<FormFieldProps> = ({ name, label, type, ...props }) => {
  const baseInputStyles = "mt-1 block w-full rounded-md border-gray-700 bg-gray-800 text-white shadow-sm focus:border-red-500 focus:ring-red-500 placeholder-gray-400";

  return (
    <div className="mb-4">
      {label && (
        <label htmlFor={name} className="block text-sm font-medium text-gray-200 mb-1">
          {label}
        </label>
      )}
      {type === 'radio' ? (
        <div className="space-y-2">
          {props.options?.map((option) => (
            <div key={option.value} className="flex items-center">
              <Field
                type="radio"
                id={`${name}-${option.value}`}
                name={name}
                value={option.value}
                className="text-red-600 bg-gray-800 border-gray-600 focus:ring-red-500"
              />
              <label htmlFor={`${name}-${option.value}`} className="ml-2 text-gray-200">
                {option.label}
              </label>
            </div>
          ))}
        </div>
      ) : type === 'checkbox' && props.multiple ? (
        <div className="space-y-2">
          {props.options?.map((option) => (
            <div key={option.value} className="flex items-center">
              <Field
                type="checkbox"
                id={`${name}-${option.value}`}
                name={name}
                value={option.value}
                className="text-red-600 bg-gray-800 border-gray-600 focus:ring-red-500"
              />
              <label htmlFor={`${name}-${option.value}`} className="ml-2 text-gray-200">
                {option.label}
              </label>
            </div>
          ))}
        </div>
      ) : type === 'number' ? (
        <Field
          id={name}
          name={name}
          type={type}
          className={baseInputStyles}
          onWheel={(e: React.WheelEvent<HTMLInputElement>) => e.currentTarget.blur()}
          {...props}
        />
      ) : (
        <Field
          id={name}
          name={name}
          type={type}
          className={baseInputStyles}
          {...props}
        />
      )}
    </div>
  );
};

export default FormField; 