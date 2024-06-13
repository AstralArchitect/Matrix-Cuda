class Matrix
{
private:
    float *matrix_d;
    int size;
public:
    Matrix(int N);
    Matrix(const Matrix& other);
    ~Matrix();

    Matrix& operator=(const Matrix& other);
    Matrix operator*(Matrix const& matrix);
    Matrix operator+(Matrix const& matrix);
    Matrix operator+(float num);
    
    float get(int x, int y);
    void set(int x, int y, float value);
};