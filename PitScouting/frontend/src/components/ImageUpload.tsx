import React, { useRef } from 'react';
import { useField } from 'formik';

interface ImageUploadProps {
  name: string;
  label: string;
  onUpload: (file: File) => void;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ name, label, onUpload }) => {
  const [field, meta, helpers] = useField(name);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onUpload(file);
      helpers.setValue(file);
    }
  };

  return (
    <div className="mt-4">
      <label className="block text-sm font-medium text-gray-200">{label}</label>
      <div className="mt-1 flex items-center">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
        >
          Choose File
        </button>
        <span className="ml-4 text-sm text-gray-400">
          {field.value ? field.value.name : 'No file chosen'}
        </span>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="image/*"
        onChange={handleFileChange}
      />
      {meta.touched && meta.error && (
        <div className="mt-1 text-sm text-red-500">{meta.error}</div>
      )}
    </div>
  );
};

export default ImageUpload; 